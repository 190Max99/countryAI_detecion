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


SCENE_CONFIGS = {
    "室内": {
        "aliases": ["室内"],
        "model_path": "models/indoor_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(10)],
        "label_names": [
            "A0_室内格局杂乱无章",
            "A1_室内家具摆放杂乱无章",
            "A2_室内生活用品杂乱无章",
            "A3_鸡鸭进入屋内共居",
            "A4_室内地面存在鸡鸭粪污",
            "A5_鸡跳在室内桌子上",
            "A6_室内地面垃圾乱丢现象严重",
            "A7_室内桌面沙发表面乱堆乱摆",
            "A8_室内墙面污迹不堪",
            "A9_室内其他脏乱情况",
        ],
        "deducts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "thresholds": [0.55, 0.55, 0.55, 0.45, 0.45, 0.45, 0.50, 0.50, 0.50, 0.65],
    },

    "庭院": {
        "aliases": ["庭院"],
        "model_path": "models/courtyard_resnet18.pth",
        "base_score": 30,
        "label_cols": [f"label_{i}" for i in range(12)],
        "label_names": [
            "B0_庭院内生产工具杂乱无章",
            "B1_庭院内交通用具杂乱无章",
            "B2_庭院内其他情况杂乱无章",
            "B3_鸡鸭进入庭院乱跑",
            "B4_庭院内鸡粪鸭粪满地",
            "B5_庭院内地面垃圾乱丢现象严重",
            "B6_庭院内柴草堆码无序",
            "B7_庭院内柴草堆码不整齐",
            "B8_庭院内房屋立面乱挂乱画",
            "B9_庭院内搭建棚库破败物品堆码杂乱不堪",
            "B10_庭院污水横流",
            "B11_庭院内其他情况",
        ],
        "deducts": [3, 3, 2, 2, 2, 5, 1, 2, 2, 5, 2, 1],
        "thresholds": [0.55, 0.55, 0.60, 0.45, 0.45, 0.50, 0.55, 0.55, 0.55, 0.55, 0.50, 0.65],
    },

    "厕所": {
        "aliases": ["厕所", "厕屋"],
        "model_path": "models/toilet_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(2)],
        "label_names": [
            "C0_厕屋脏乱",
            "C1_厕屋功能配备不齐全",
        ],
        "deducts": [2, 3],
        "thresholds": [0.50, 0.50],
    },

    "化粪池": {
        "aliases": ["化粪池"],
        "model_path": "models/septic_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(3)],
        "label_names": [
            "C0_化粪池盖板挪开未关闭取粪口未关闭",
            "C1_化粪池粪污溢流",
            "C2_厕所周围其他情况",
        ],
        "deducts": [2, 2, 1],
        "thresholds": [0.50, 0.50, 0.65],
    },

    "房前屋后": {
        "aliases": ["房前屋后", "房屋前后", "房前屋后及2侧", "房前屋后及两侧", "屋后", "两侧", "2侧"],
        "model_path": "models/outside_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(5)],
        "label_names": [
            "D0_房屋旁柴草堆码乱堆不整齐",
            "D1_房屋周身存在污水横流现象",
            "D2_房屋周身瓜果棚架破败不堪",
            "D3_房屋周身鸡鸭棚圈破败不堪脏臭",
            "D4_房屋周身其他情况",
        ],
        "deducts": [3, 2, 2, 2, 1],
        "thresholds": [0.55, 0.50, 0.50, 0.50, 0.65],
    },
}


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_csv_safely(csv_path: Path):
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            print(f"读取人工标签CSV成功: {csv_path}，编码: {enc}")
            return df
        except Exception as e:
            last_error = e

    raise last_error


def find_label_csv(folder: Path):
    csv_files = sorted(folder.glob("*.csv"))

    if len(csv_files) == 0:
        return None

    return csv_files[0]


def normalize_scene_name(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def prepare_local_label_df(label_csv: Path):
    """
    兼容两种格式：

    格式1：
    scene,image_path,label_0,label_1,...

    格式2：
    序号,label_0,label_1,...
    厕所,1,0,...
    房前屋后,0,0,...
    室内,1,1,...
    """
    df = read_csv_safely(label_csv)
    df.columns = [str(c).strip() for c in df.columns]

    if "scene" in df.columns:
        scene_col = "scene"
    elif "序号" in df.columns:
        scene_col = "序号"
    elif "类别" in df.columns:
        scene_col = "类别"
    else:
        scene_col = df.columns[0]

    df["scene_norm"] = df[scene_col].apply(normalize_scene_name)

    return df


def find_true_row(local_df: pd.DataFrame, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    rows = local_df[local_df["scene_norm"].isin(aliases)]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


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


def get_true_labels(row, label_cols):
    labels = []

    for col in label_cols:
        labels.append(get_label_value(row, col))

    return np.array(labels, dtype=int)


def calc_deduct(labels, deducts):
    total = 0

    for flag, deduct in zip(labels, deducts):
        if int(flag) == 1:
            total += int(deduct)

    return total


def find_image_in_folder(folder: Path, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    candidates = []

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        filename = file.name

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


def get_predict_transform():
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


def predict_scene_image(scene_key, image_path, device):
    cfg = SCENE_CONFIGS[scene_key]

    model_path = Path(cfg["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"{scene_key} 模型不存在: {model_path}")

    checkpoint = load_checkpoint(model_path, device)

    label_names = checkpoint.get("label_names", cfg["label_names"])
    deducts = checkpoint.get("deducts", cfg["deducts"])
    thresholds = np.array(checkpoint.get("thresholds", cfg["thresholds"]), dtype=float)

    model = build_model(num_labels=len(label_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_predict_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pred_labels = (probs >= thresholds).astype(int)
    pred_deduct = calc_deduct(pred_labels, deducts)

    return {
        "label_names": label_names,
        "deducts": deducts,
        "thresholds": thresholds,
        "probs": probs,
        "pred_labels": pred_labels,
        "pred_deduct": pred_deduct,
    }


def normalize_house_id_from_folder(folder: Path):
    return folder.name.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="某一户图片和CSV所在文件夹，例如 data/raw/61")
    parser.add_argument("--label_csv", default=None, help="该户人工标签CSV；不填则自动读取文件夹中的第一个CSV")
    parser.add_argument("--house_id", default=None, help="农户编号；不填则使用文件夹名")
    parser.add_argument("--out", default=None, help="输出汇总CSV路径")
    parser.add_argument("--detail_out", default=None, help="输出标签详细CSV路径")
    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        raise FileNotFoundError(f"找不到文件夹: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {folder}")

    if args.label_csv is None:
        label_csv = find_label_csv(folder)
    else:
        label_csv = Path(args.label_csv)

    if label_csv is None:
        raise FileNotFoundError(f"文件夹中没有找到CSV文件: {folder}")

    if not label_csv.exists():
        raise FileNotFoundError(f"找不到人工标签CSV: {label_csv}")

    house_id = args.house_id if args.house_id is not None else normalize_house_id_from_folder(folder)

    local_df = prepare_local_label_df(label_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==============================")
    print("农户文件夹整体评分")
    print("图片文件夹:", folder)
    print("人工标签CSV:", label_csv)
    print("house_id:", house_id)
    print("当前设备:", device)
    print("==============================")

    scene_results = {}
    detail_rows = []

    for scene_key in ["室内", "庭院", "厕所", "化粪池", "房前屋后"]:
        cfg = SCENE_CONFIGS[scene_key]

        print(f"\n正在处理场景：{scene_key}")

        image_path = find_image_in_folder(folder, scene_key)
        true_row = find_true_row(local_df, scene_key)

        if true_row is None:
            print(f"人工CSV中没有找到 {scene_key} 标签")
            true_labels = None
            true_deduct = np.nan
        else:
            true_labels = get_true_labels(true_row, cfg["label_cols"])
            true_deduct = calc_deduct(true_labels, cfg["deducts"])
            print("人工扣分:", true_deduct)

        if image_path is None:
            print(f"文件夹中没有找到 {scene_key} 图片")
            pred_labels = None
            probs = None
            thresholds = np.array(cfg["thresholds"], dtype=float)
            label_names = cfg["label_names"]
            deducts = cfg["deducts"]
            pred_deduct = np.nan
        else:
            print("找到图片:", image_path)

            pred_result = predict_scene_image(scene_key, image_path, device)

            pred_labels = pred_result["pred_labels"]
            probs = pred_result["probs"]
            thresholds = pred_result["thresholds"]
            label_names = pred_result["label_names"]
            deducts = pred_result["deducts"]
            pred_deduct = pred_result["pred_deduct"]

            print("模型扣分:", pred_deduct)

        scene_results[scene_key] = {
            "scene": scene_key,
            "image_path": str(image_path) if image_path is not None else "",
            "true_deduct": true_deduct,
            "pred_deduct": pred_deduct,
        }

        for i, label_name in enumerate(label_names):
            true_label = np.nan
            pred_label = np.nan
            prob = np.nan
            threshold = float(thresholds[i])
            deduct_value = deducts[i]

            if true_labels is not None and i < len(true_labels):
                true_label = int(true_labels[i])

            if pred_labels is not None and i < len(pred_labels):
                pred_label = int(pred_labels[i])
                prob = float(probs[i])

            is_correct = np.nan
            if not pd.isna(true_label) and not pd.isna(pred_label):
                is_correct = int(true_label == pred_label)

            detail_rows.append({
                "house_id": house_id,
                "scene": scene_key,
                "image_path": str(image_path) if image_path is not None else "",
                "label": f"label_{i}",
                "label_name": label_name,
                "deduct_value": deduct_value,
                "true_label": true_label,
                "pred_label": pred_label,
                "prob": prob,
                "threshold": threshold,
                "is_correct": is_correct,
            })

            mark = ""
            if is_correct == 1:
                mark = "√"
            elif is_correct == 0:
                mark = "×"

            prob_text = "nan" if pd.isna(prob) else f"{prob:.3f}"

            print(
                f"  {mark} label_{i} {label_name} | "
                f"人工={true_label} | "
                f"模型={pred_label} | "
                f"概率={prob_text} | "
                f"阈值={threshold:.2f} | "
                f"扣分值={deduct_value}"
            )

    summary_rows = []

    total_true_score = 0
    total_pred_score = 0

    # 室内 10分
    indoor = scene_results.get("室内")
    if indoor is not None:
        base = 10
        true_score = np.nan if pd.isna(indoor["true_deduct"]) else max(0, base - indoor["true_deduct"])
        pred_score = np.nan if pd.isna(indoor["pred_deduct"]) else max(0, base - indoor["pred_deduct"])

        if not pd.isna(true_score):
            total_true_score += true_score
        if not pd.isna(pred_score):
            total_pred_score += pred_score

        summary_rows.append({
            "house_id": house_id,
            "category": "室内",
            "base_score": base,
            "true_deduct": indoor["true_deduct"],
            "pred_deduct": indoor["pred_deduct"],
            "true_score": true_score,
            "pred_score": pred_score,
            "score_error": pred_score - true_score if not pd.isna(true_score) and not pd.isna(pred_score) else np.nan,
        })

    # 庭院 30分
    courtyard = scene_results.get("庭院")
    if courtyard is not None:
        base = 30
        true_score = np.nan if pd.isna(courtyard["true_deduct"]) else max(0, base - courtyard["true_deduct"])
        pred_score = np.nan if pd.isna(courtyard["pred_deduct"]) else max(0, base - courtyard["pred_deduct"])

        if not pd.isna(true_score):
            total_true_score += true_score
        if not pd.isna(pred_score):
            total_pred_score += pred_score

        summary_rows.append({
            "house_id": house_id,
            "category": "庭院",
            "base_score": base,
            "true_deduct": courtyard["true_deduct"],
            "pred_deduct": courtyard["pred_deduct"],
            "true_score": true_score,
            "pred_score": pred_score,
            "score_error": pred_score - true_score if not pd.isna(true_score) and not pd.isna(pred_score) else np.nan,
        })

    # 厕所及化粪池 10分
    toilet_true_deduct = 0
    toilet_pred_deduct = 0
    has_true_toilet_part = False
    has_pred_toilet_part = False

    for scene_key in ["厕所", "化粪池"]:
        r = scene_results.get(scene_key)

        if r is None:
            continue

        if not pd.isna(r["true_deduct"]):
            toilet_true_deduct += r["true_deduct"]
            has_true_toilet_part = True

        if not pd.isna(r["pred_deduct"]):
            toilet_pred_deduct += r["pred_deduct"]
            has_pred_toilet_part = True

    base = 10

    true_score = np.nan
    pred_score = np.nan

    if has_true_toilet_part:
        true_score = max(0, base - toilet_true_deduct)
        total_true_score += true_score

    if has_pred_toilet_part:
        pred_score = max(0, base - toilet_pred_deduct)
        total_pred_score += pred_score

    summary_rows.append({
        "house_id": house_id,
        "category": "厕所及化粪池",
        "base_score": base,
        "true_deduct": toilet_true_deduct if has_true_toilet_part else np.nan,
        "pred_deduct": toilet_pred_deduct if has_pred_toilet_part else np.nan,
        "true_score": true_score,
        "pred_score": pred_score,
        "score_error": pred_score - true_score if not pd.isna(true_score) and not pd.isna(pred_score) else np.nan,
    })

    # 房前屋后 10分
    outside = scene_results.get("房前屋后")
    if outside is not None:
        base = 10
        true_score = np.nan if pd.isna(outside["true_deduct"]) else max(0, base - outside["true_deduct"])
        pred_score = np.nan if pd.isna(outside["pred_deduct"]) else max(0, base - outside["pred_deduct"])

        if not pd.isna(true_score):
            total_true_score += true_score
        if not pd.isna(pred_score):
            total_pred_score += pred_score

        summary_rows.append({
            "house_id": house_id,
            "category": "房前屋后",
            "base_score": base,
            "true_deduct": outside["true_deduct"],
            "pred_deduct": outside["pred_deduct"],
            "true_score": true_score,
            "pred_score": pred_score,
            "score_error": pred_score - true_score if not pd.isna(true_score) and not pd.isna(pred_score) else np.nan,
        })

    summary_rows.append({
        "house_id": house_id,
        "category": "总分",
        "base_score": 60,
        "true_deduct": 60 - total_true_score,
        "pred_deduct": 60 - total_pred_score,
        "true_score": total_true_score,
        "pred_score": total_pred_score,
        "score_error": total_pred_score - total_true_score,
    })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    print("\n\n========== 汇总评分对比 ==========")
    print(summary_df.to_string(index=False))

    print("\n========== 最终结果 ==========")
    print(f"人工总分: {total_true_score} / 60")
    print(f"模型总分: {total_pred_score} / 60")
    print(f"总分误差: {total_pred_score - total_true_score}")

    if args.out is None:
        out_path = csv_output_path(f"folder_score_result_{house_id}.csv")
    else:
        out_path = csv_output_path(args.out)

    if args.detail_out is None:
        detail_out_path = csv_output_path(f"folder_score_detail_{house_id}.csv")
    else:
        detail_out_path = csv_output_path(args.detail_out)

    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_out_path, index=False, encoding="utf-8-sig")

    print("\n汇总结果已保存:", out_path)
    print("标签详细结果已保存:", detail_out_path)


if __name__ == "__main__":
    main()

