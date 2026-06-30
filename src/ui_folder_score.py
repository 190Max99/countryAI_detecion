import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================
# 1. 场景配置
# =========================

SCENE_CONFIGS = {
    "室内": {
        "aliases": ["室内"],
        "model_path": "models/indoor_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(10)],
        "deducts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "thresholds": [0.55, 0.55, 0.55, 0.45, 0.45, 0.45, 0.50, 0.50, 0.50, 0.65],
    },

    "庭院": {
        "aliases": ["庭院"],
        "model_path": "models/courtyard_resnet18.pth",
        "base_score": 30,
        "label_cols": [f"label_{i}" for i in range(12)],
        "deducts": [3, 3, 2, 2, 2, 5, 1, 2, 2, 5, 2, 1],
        "thresholds": [0.55, 0.55, 0.60, 0.45, 0.45, 0.50, 0.55, 0.55, 0.55, 0.55, 0.50, 0.65],
    },

    "厕所": {
        "aliases": ["厕所", "厕屋"],
        "model_path": "models/toilet_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(2)],
        "deducts": [2, 3],
        "thresholds": [0.50, 0.50],
    },

    "化粪池": {
        "aliases": ["化粪池"],
        "model_path": "models/septic_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(3)],
        "deducts": [2, 2, 1],
        "thresholds": [0.50, 0.50, 0.65],
    },

    "房前屋后": {
        "aliases": ["房前屋后", "房屋前后", "房前屋后及2侧", "房前屋后及两侧", "屋后", "两侧", "2侧"],
        "model_path": "models/outside_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(5)],
        "deducts": [3, 2, 2, 2, 1],
        "thresholds": [0.55, 0.50, 0.50, 0.50, 0.65],
    },
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
MODEL_CACHE = {}


# =========================
# 2. CSV 与分数计算
# =========================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("\t", "")
    text = text.replace(" ", "")
    text = text.replace("\u3000", "")
    text = text.strip()

    return text


def read_csv_safely(csv_path: Path):
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    seps = [None, "\t", ","]

    best_df = None
    best_col_count = 0

    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(csv_path, encoding=enc, sep=sep)

                df.columns = [normalize_text(c) for c in df.columns]

                if len(df.columns) > best_col_count:
                    best_df = df
                    best_col_count = len(df.columns)

                if len(df.columns) >= 3:
                    return df

            except Exception:
                pass

    if best_df is not None:
        return best_df

    raise RuntimeError(f"无法读取CSV文件: {csv_path}")


def find_label_csv(folder: Path):
    preferred = folder / f"{folder.name}.csv"

    if preferred.exists():
        return preferred

    csv_files = sorted(folder.glob("*.csv"))

    bad_keywords = ["result", "detail", "score", "预测", "结果", "simple"]

    valid_files = []

    for file in csv_files:
        name = file.name.lower()
        if any(k.lower() in name for k in bad_keywords):
            continue
        valid_files.append(file)

    if len(valid_files) > 0:
        return valid_files[0]

    if len(csv_files) > 0:
        return csv_files[0]

    return None


def prepare_label_df(label_csv: Path):
    df = read_csv_safely(label_csv)

    if "scene" in df.columns:
        scene_col = "scene"
    elif "序号" in df.columns:
        scene_col = "序号"
    elif "类别" in df.columns:
        scene_col = "类别"
    else:
        scene_col = df.columns[0]

    df["scene_norm"] = df[scene_col].apply(normalize_text)

    return df


def find_true_row(label_df: pd.DataFrame, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    rows = label_df[label_df["scene_norm"].isin(aliases)]

    if len(rows) > 0:
        return rows.iloc[0]

    for _, row in label_df.iterrows():
        scene_text = str(row["scene_norm"])

        for alias in aliases:
            if alias in scene_text:
                return row

    return None


def get_label_value(row, col):
    if row is None:
        return 0

    if col not in row.index:
        return 0

    value = row[col]

    if pd.isna(value):
        return 0

    try:
        return int(float(value))
    except Exception:
        return 0


def calc_deduct_from_row(row, scene_key):
    cfg = SCENE_CONFIGS[scene_key]

    total_deduct = 0

    for col, deduct in zip(cfg["label_cols"], cfg["deducts"]):
        flag = get_label_value(row, col)

        if flag == 1:
            total_deduct += deduct

    return total_deduct


def calc_score(base_score, deduct):
    if pd.isna(deduct):
        return np.nan

    return max(0, base_score - deduct)


def calc_score_error(true_score, pred_score):
    if pd.isna(true_score) or pd.isna(pred_score):
        return np.nan

    return pred_score - true_score


# =========================
# 3. 图片与模型预测
# =========================

def find_image_in_folder(folder: Path, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    candidates = []

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        filename = normalize_text(file.name)

        for alias in aliases:
            if alias in filename:
                candidates.append(file)
                break

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: p.name)

    return candidates[0]


def build_model(num_labels):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_model_for_scene(scene_key, device):
    if scene_key in MODEL_CACHE:
        return MODEL_CACHE[scene_key]

    cfg = SCENE_CONFIGS[scene_key]
    model_path = Path(cfg["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"{scene_key} 模型不存在: {model_path}")

    checkpoint = load_checkpoint(model_path, device)

    deducts = checkpoint.get("deducts", cfg["deducts"])
    thresholds = np.array(checkpoint.get("thresholds", cfg["thresholds"]), dtype=float)
    label_names = checkpoint.get("label_names", cfg["label_cols"])

    model = build_model(len(label_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    MODEL_CACHE[scene_key] = {
        "model": model,
        "deducts": deducts,
        "thresholds": thresholds,
    }

    return MODEL_CACHE[scene_key]


def predict_scene_deduct(scene_key, image_path, device):
    model_pack = load_model_for_scene(scene_key, device)

    model = model_pack["model"]
    deducts = model_pack["deducts"]
    thresholds = model_pack["thresholds"]

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pred_labels = (probs >= thresholds).astype(int)

    total_deduct = 0

    for flag, deduct in zip(pred_labels, deducts):
        if int(flag) == 1:
            total_deduct += int(deduct)

    return total_deduct


# =========================
# 4. 核心评分函数
# =========================

def score_folder(folder: Path):
    label_csv = find_label_csv(folder)

    if label_csv is None:
        raise FileNotFoundError(f"文件夹中没有找到人工标签CSV: {folder}")

    label_df = prepare_label_df(label_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene_deducts = {}

    for scene_key in ["室内", "庭院", "厕所", "化粪池", "房前屋后"]:
        true_row = find_true_row(label_df, scene_key)
        image_path = find_image_in_folder(folder, scene_key)

        if true_row is None:
            true_deduct = np.nan
        else:
            true_deduct = calc_deduct_from_row(true_row, scene_key)

        if image_path is None:
            pred_deduct = np.nan
        else:
            pred_deduct = predict_scene_deduct(scene_key, image_path, device)

        scene_deducts[scene_key] = {
            "true_deduct": true_deduct,
            "pred_deduct": pred_deduct,
            "image_path": str(image_path) if image_path is not None else "",
        }

    rows = []

    # 室内
    indoor_true_deduct = scene_deducts["室内"]["true_deduct"]
    indoor_pred_deduct = scene_deducts["室内"]["pred_deduct"]

    indoor_true_score = calc_score(10, indoor_true_deduct)
    indoor_pred_score = calc_score(10, indoor_pred_deduct)

    rows.append({
        "项目": "室内",
        "满分": 10,
        "实际扣分": indoor_true_deduct,
        "预测扣分": indoor_pred_deduct,
        "实际得分": indoor_true_score,
        "预测得分": indoor_pred_score,
        "分差": calc_score_error(indoor_true_score, indoor_pred_score),
    })

    # 庭院
    courtyard_true_deduct = scene_deducts["庭院"]["true_deduct"]
    courtyard_pred_deduct = scene_deducts["庭院"]["pred_deduct"]

    courtyard_true_score = calc_score(30, courtyard_true_deduct)
    courtyard_pred_score = calc_score(30, courtyard_pred_deduct)

    rows.append({
        "项目": "庭院",
        "满分": 30,
        "实际扣分": courtyard_true_deduct,
        "预测扣分": courtyard_pred_deduct,
        "实际得分": courtyard_true_score,
        "预测得分": courtyard_pred_score,
        "分差": calc_score_error(courtyard_true_score, courtyard_pred_score),
    })

    # 厕所及化粪池
    toilet_true = scene_deducts["厕所"]["true_deduct"]
    septic_true = scene_deducts["化粪池"]["true_deduct"]

    toilet_pred = scene_deducts["厕所"]["pred_deduct"]
    septic_pred = scene_deducts["化粪池"]["pred_deduct"]

    if pd.isna(toilet_true) or pd.isna(septic_true):
        toilet_septic_true_deduct = np.nan
    else:
        toilet_septic_true_deduct = toilet_true + septic_true

    if pd.isna(toilet_pred) or pd.isna(septic_pred):
        toilet_septic_pred_deduct = np.nan
    else:
        toilet_septic_pred_deduct = toilet_pred + septic_pred

    toilet_septic_true_score = calc_score(10, toilet_septic_true_deduct)
    toilet_septic_pred_score = calc_score(10, toilet_septic_pred_deduct)

    rows.append({
        "项目": "厕所及化粪池",
        "满分": 10,
        "实际扣分": toilet_septic_true_deduct,
        "预测扣分": toilet_septic_pred_deduct,
        "实际得分": toilet_septic_true_score,
        "预测得分": toilet_septic_pred_score,
        "分差": calc_score_error(toilet_septic_true_score, toilet_septic_pred_score),
    })

    # 房前屋后
    outside_true_deduct = scene_deducts["房前屋后"]["true_deduct"]
    outside_pred_deduct = scene_deducts["房前屋后"]["pred_deduct"]

    outside_true_score = calc_score(10, outside_true_deduct)
    outside_pred_score = calc_score(10, outside_pred_deduct)

    rows.append({
        "项目": "房前屋后",
        "满分": 10,
        "实际扣分": outside_true_deduct,
        "预测扣分": outside_pred_deduct,
        "实际得分": outside_true_score,
        "预测得分": outside_pred_score,
        "分差": calc_score_error(outside_true_score, outside_pred_score),
    })

    result_df = pd.DataFrame(rows)

    true_scores = result_df["实际得分"].tolist()
    pred_scores = result_df["预测得分"].tolist()

    if any(pd.isna(x) for x in true_scores):
        total_true_score = np.nan
    else:
        total_true_score = sum(true_scores)

    if any(pd.isna(x) for x in pred_scores):
        total_pred_score = np.nan
    else:
        total_pred_score = sum(pred_scores)

    result_df.loc[len(result_df)] = {
        "项目": "总分",
        "满分": 60,
        "实际扣分": 60 - total_true_score if not pd.isna(total_true_score) else np.nan,
        "预测扣分": 60 - total_pred_score if not pd.isna(total_pred_score) else np.nan,
        "实际得分": total_true_score,
        "预测得分": total_pred_score,
        "分差": calc_score_error(total_true_score, total_pred_score),
    }

    save_path = folder / f"ui_score_result_{folder.name}.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result_df, label_csv, save_path, device


# =========================
# 5. UI 界面
# =========================

class ScoreApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI积分制现场照片评分系统")
        self.root.geometry("980x560")

        self.selected_folder = None

        title = tk.Label(
            root,
            text="AI积分制现场照片评分系统",
            font=("Microsoft YaHei", 18, "bold")
        )
        title.pack(pady=12)

        top_frame = tk.Frame(root)
        top_frame.pack(fill="x", padx=20)

        self.folder_label = tk.Label(
            top_frame,
            text="当前未选择文件夹",
            anchor="w",
            font=("Microsoft YaHei", 10)
        )
        self.folder_label.pack(side="left", fill="x", expand=True)

        choose_btn = tk.Button(
            top_frame,
            text="选择农户文件夹",
            command=self.choose_folder,
            width=16,
            font=("Microsoft YaHei", 10)
        )
        choose_btn.pack(side="right", padx=5)

        run_btn = tk.Button(
            top_frame,
            text="开始评分",
            command=self.run_score_thread,
            width=12,
            font=("Microsoft YaHei", 10)
        )
        run_btn.pack(side="right", padx=5)

        columns = ["项目", "满分", "实际扣分", "预测扣分", "实际得分", "预测得分", "分差"]

        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=8)
        self.tree.pack(fill="both", expand=True, padx=20, pady=16)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=120)

        self.status_text = tk.Text(root, height=8, font=("Consolas", 10))
        self.status_text.pack(fill="x", padx=20, pady=8)

        self.log("请选择一个农户文件夹，例如 data/raw/97。")

    def log(self, message):
        self.status_text.insert("end", str(message) + "\n")
        self.status_text.see("end")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="选择农户文件夹")

        if folder:
            self.selected_folder = Path(folder)
            self.folder_label.config(text=f"当前文件夹：{self.selected_folder}")
            self.log(f"已选择文件夹：{self.selected_folder}")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def run_score_thread(self):
        if self.selected_folder is None:
            messagebox.showwarning("提示", "请先选择农户文件夹")
            return

        thread = threading.Thread(target=self.run_score, daemon=True)
        thread.start()

    def format_value(self, value):
        if pd.isna(value):
            return "缺失"

        try:
            if float(value).is_integer():
                return str(int(value))
        except Exception:
            pass

        return str(value)

    def run_score(self):
        try:
            self.clear_table()
            self.log("开始评分，请稍等...")

            result_df, label_csv, save_path, device = score_folder(self.selected_folder)

            self.log(f"人工标签CSV：{label_csv}")
            self.log(f"运行设备：{device}")

            for _, row in result_df.iterrows():
                values = [
                    row["项目"],
                    self.format_value(row["满分"]),
                    self.format_value(row["实际扣分"]),
                    self.format_value(row["预测扣分"]),
                    self.format_value(row["实际得分"]),
                    self.format_value(row["预测得分"]),
                    self.format_value(row["分差"]),
                ]

                self.tree.insert("", "end", values=values)

            total_row = result_df[result_df["项目"] == "总分"].iloc[0]

            self.log("评分完成。")
            self.log(f"实际总分：{self.format_value(total_row['实际得分'])} / 60")
            self.log(f"预测总分：{self.format_value(total_row['预测得分'])} / 60")
            self.log(f"总分分差：{self.format_value(total_row['分差'])}")
            self.log(f"结果已保存：{save_path}")

        except Exception as e:
            self.log(f"错误：{e}")
            messagebox.showerror("运行错误", str(e))


def main():
    root = tk.Tk()
    app = ScoreApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()