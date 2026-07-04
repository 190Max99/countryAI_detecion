import argparse
from pathlib import Path

try:
    from src.output_utils import csv_output_path
except ModuleNotFoundError:
    from output_utils import csv_output_path

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


# =========================
# 2. 基础工具函数
# =========================

def normalize_text(value):
    """
    去掉中文 CSV 中可能出现的隐藏字符、空格、制表符。
    """
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
    """
    兼容：
    1. utf-8-sig / gbk / gb18030 / utf-8
    2. 逗号分隔
    3. 制表符分隔
    """
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    seps = [None, "\t", ","]

    best_df = None
    best_info = None
    best_col_count = 0

    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
                    sep_name = "auto"
                else:
                    df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                    sep_name = repr(sep)

                df.columns = [normalize_text(c) for c in df.columns]

                if len(df.columns) > best_col_count:
                    best_df = df
                    best_info = (enc, sep_name)
                    best_col_count = len(df.columns)

                if len(df.columns) >= 3:
                    print(f"人工CSV读取成功: {csv_path}")
                    print(f"编码: {enc}, 分隔符: {sep_name}")
                    return df

            except Exception:
                pass

    if best_df is not None:
        print(f"人工CSV读取成功: {csv_path}")
        print(f"编码: {best_info[0]}, 分隔符: {best_info[1]}")
        return best_df

    raise RuntimeError(f"无法读取CSV文件: {csv_path}")


def find_label_csv(folder: Path):
    """
    自动找文件夹里的人工标签 CSV。
    优先找：文件夹名.csv，例如 97.csv。
    避免误读输出结果文件。
    """
    preferred = folder / f"{folder.name}.csv"

    if preferred.exists():
        return preferred

    csv_files = sorted(folder.glob("*.csv"))

    bad_keywords = ["result", "detail", "score", "预测", "结果"]

    valid_csv_files = []

    for file in csv_files:
        name = file.name.lower()

        if any(k.lower() in name for k in bad_keywords):
            continue

        valid_csv_files.append(file)

    if len(valid_csv_files) > 0:
        return valid_csv_files[0]

    if len(csv_files) > 0:
        return csv_files[0]

    return None


def prepare_label_df(label_csv: Path):
    """
    读取人工标签 CSV，并识别场景列。
    支持：
    序号,label_0,label_1,...
    scene,label_0,label_1,...
    类别,label_0,label_1,...
    """
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

    print("识别到的场景值:", df["scene_norm"].tolist())

    return df


def find_true_row(label_df: pd.DataFrame, scene_key: str):
    """
    从人工 CSV 中找到某个场景的那一行。
    先精确匹配，再模糊匹配。
    """
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
    """
    读取 label_0、label_1 等。
    空值或异常值默认按 0 处理。
    """
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
    """
    根据人工标签行计算该场景扣分。
    """
    cfg = SCENE_CONFIGS[scene_key]

    total_deduct = 0

    for col, deduct in zip(cfg["label_cols"], cfg["deducts"]):
        flag = get_label_value(row, col)

        if flag == 1:
            total_deduct += deduct

    return total_deduct


def calc_score(base_score, deduct):
    """
    根据满分和扣分计算得分。
    """
    if pd.isna(deduct):
        return np.nan

    return max(0, base_score - deduct)


def calc_score_error(true_score, pred_score):
    """
    分差 = 预测得分 - 实际得分
    分差 > 0：模型给分偏高
    分差 < 0：模型给分偏低
    """
    if pd.isna(true_score) or pd.isna(pred_score):
        return np.nan

    return pred_score - true_score


# =========================
# 3. 图片与模型预测
# =========================

def find_image_in_folder(folder: Path, scene_key: str):
    """
    在文件夹中自动找对应场景图片。
    """
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


def predict_scene_deduct(scene_key, image_path, device):
    """
    对某个场景的一张图片进行预测，并返回扣分。
    """
    cfg = SCENE_CONFIGS[scene_key]

    model_path = Path(cfg["model_path"])

    if not model_path.exists():
        print(f"警告：{scene_key} 模型不存在: {model_path}")
        return np.nan

    checkpoint = load_checkpoint(model_path, device)

    deducts = checkpoint.get("deducts", cfg["deducts"])
    thresholds = np.array(checkpoint.get("thresholds", cfg["thresholds"]), dtype=float)

    label_names = checkpoint.get("label_names", cfg["label_cols"])
    num_labels = len(label_names)

    model = build_model(num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
# 4. 主程序
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="某户文件夹，例如 data/raw/97")
    parser.add_argument("--label_csv", default=None, help="人工标签CSV，不填则自动找文件夹里的CSV")
    parser.add_argument("--out", default=None, help="输出CSV，不填则自动命名")
    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        raise FileNotFoundError(f"找不到文件夹: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"输入的不是文件夹: {folder}")

    if args.label_csv is None:
        label_csv = find_label_csv(folder)
    else:
        label_csv = Path(args.label_csv)

    if label_csv is None:
        raise FileNotFoundError(f"文件夹中没有找到人工标签CSV: {folder}")

    if not label_csv.exists():
        raise FileNotFoundError(f"找不到人工标签CSV: {label_csv}")

    house_id = folder.name

    label_df = prepare_label_df(label_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print("农户文件夹评分")
    print("house_id:", house_id)
    print("文件夹:", folder)
    print("人工标签CSV:", label_csv)
    print("设备:", device)
    print("==============================")

    # =========================
    # 逐场景计算实际扣分和预测扣分
    # =========================

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
        }

    # =========================
    # 汇总成四大项
    # =========================

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

    # =========================
    # 总分
    # =========================

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

    # =========================
    # 输出
    # =========================

    print("\n========== 评分结果 ==========")
    print(result_df.to_string(index=False))

    if not pd.isna(total_true_score) and not pd.isna(total_pred_score):
        print("\n实际总分:", int(total_true_score), "/ 60")
        print("预测总分:", int(total_pred_score), "/ 60")
        print("总分分差:", int(total_pred_score - total_true_score))
    else:
        print("\n注意：存在缺失的人工标签或图片，无法完整计算总分。")

    if args.out is None:
        out_path = csv_output_path(f"folder_score_simple_{house_id}.csv")
    else:
        out_path = csv_output_path(args.out)

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存:", out_path)


if __name__ == "__main__":
    main()

