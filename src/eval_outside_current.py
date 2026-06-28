import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.metrics import f1_score, precision_score, recall_score


SCENE_NAMES = ["房前屋后", "房屋前后"]

LABEL_COLS = [f"label_{i}" for i in range(5)]


def read_csv_safely(csv_path: Path):
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    last_error = None

    for enc in encodings:
        try:
            print(f"尝试使用编码读取 CSV: {enc}")
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            print(f"读取成功，使用编码: {enc}")
            return df
        except Exception as e:
            last_error = e
            print(f"编码 {enc} 读取失败: {e}")

    raise last_error


def build_model(num_labels=5):
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


def calc_score(labels, deducts):
    total_deduct = 0

    for flag, deduct in zip(labels, deducts):
        if int(flag) == 1:
            total_deduct += int(deduct)

    final_score = max(0, 10 - total_deduct)
    return final_score, total_deduct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--model", default="models/outside_resnet18.pth")
    parser.add_argument("--out", default="outside_eval_current_result.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    df = read_csv_safely(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["scene", "image_path"] + LABEL_COLS

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中找不到列: {col}")

    if "house_id" not in df.columns:
        df["house_id"] = range(len(df))

    df["scene"] = df["scene"].astype(str).str.strip()

    outside_df = df[df["scene"].isin(SCENE_NAMES)].copy()

    if len(outside_df) == 0:
        print("当前 CSV 中 scene 的取值：")
        print(df["scene"].unique())
        raise ValueError("没有找到 scene == 房前屋后 或 房屋前后 的数据")

    for col in LABEL_COLS:
        outside_df[col] = outside_df[col].fillna(0).astype(int)

    outside_df["image_path"] = outside_df["image_path"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n当前设备:", device)

    checkpoint = load_checkpoint(model_path, device)

    label_names = checkpoint.get("label_names", [
        "D0_房屋旁柴草堆码乱堆不整齐",
        "D1_房屋周身存在污水横流现象",
        "D2_房屋周身瓜果棚架破败不堪",
        "D3_房屋周身鸡鸭棚圈破败不堪脏臭",
        "D4_房屋周身其他情况",
    ])

    deducts = checkpoint.get("deducts", [3, 2, 2, 2, 1])

    thresholds = np.array(checkpoint.get("thresholds", [
        0.55,
        0.50,
        0.50,
        0.50,
        0.65,
    ]))

    model = build_model(num_labels=len(label_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tf = get_transform()

    all_true = []
    all_pred = []
    all_prob = []
    result_rows = []

    for _, row in outside_df.iterrows():
        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            print("图片不存在，跳过:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = tf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        true_labels = row[LABEL_COLS].values.astype(int)
        pred_labels = (probs >= thresholds).astype(int)

        true_score, true_deduct = calc_score(true_labels, deducts)
        pred_score, pred_deduct = calc_score(pred_labels, deducts)

        result = {
            "house_id": row.get("house_id", ""),
            "image_path": str(image_path),
            "人工扣分": true_deduct,
            "人工得分": true_score,
            "模型扣分": pred_deduct,
            "模型得分": pred_score,
            "得分误差": abs(true_score - pred_score),
            "标签完全一致": int(np.array_equal(true_labels, pred_labels)),
        }

        for i in range(len(label_names)):
            result[f"true_label_{i}"] = int(true_labels[i])
            result[f"prob_label_{i}"] = round(float(probs[i]), 4)
            result[f"pred_label_{i}"] = int(pred_labels[i])

        result_rows.append(result)
        all_true.append(true_labels)
        all_pred.append(pred_labels)
        all_prob.append(probs)

    if len(result_rows) == 0:
        raise ValueError("没有成功测评任何图片，请检查 image_path 是否正确")

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    label_acc = (y_true == y_pred).mean()
    exact_match = np.mean([np.array_equal(t, p) for t, p in zip(y_true, y_pred)])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    mean_score_error = result_df["得分误差"].mean()

    print("\n========== 房前屋后模型重新测评结果 ==========")
    print("测评图片数量:", len(result_df))
    print("标签平均准确率:", round(label_acc, 4))
    print("整张图片标签完全一致率:", round(exact_match, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print("Macro Recall:", round(macro_recall, 4))
    print("平均得分误差:", round(mean_score_error, 4))
    print("详细结果已保存:", args.out)

    print("\n========== 各标签 F1 ==========")
    for i, name in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        pos_count = int(y_true[:, i].sum())
        pred_count = int(y_pred[:, i].sum())

        print(
            f"label_{i} {name}: "
            f"F1={f1:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"人工正样本={pos_count}, "
            f"模型预测正样本={pred_count}"
        )

    print("\n========== 得分误差分布 ==========")
    print(result_df["得分误差"].value_counts().sort_index())

    print("\n测评完成。")


if __name__ == "__main__":
    main()