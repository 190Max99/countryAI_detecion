"""
房前屋后模型 holdout 验证脚本。
读取 outside_holdout_rows.csv（训练时保留的验证数据），用 outside_resnet18.pth 逐张推理，
对比人工标注与模型预测，输出：标签准确率、F1/Precision/Recall、得分误差等。

用法: python eval_outside_holdout.py [--holdout <holdout CSV>] [--model <模型路径>] [--out <输出CSV>]
"""

import argparse
from pathlib import Path

try:
    from src.output_utils import csv_input_path, csv_output_path
except ModuleNotFoundError:
    from output_utils import csv_input_path, csv_output_path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.metrics import f1_score, precision_score, recall_score


# 房前屋后场景有 5 个标签列
LABEL_COLS = [f"label_{i}" for i in range(5)]


def build_model(num_labels=5):
    """构建 ResNet18 多标签分类模型（房前屋后5个标签）。"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
    return model


def get_transform():
    """图像预处理：缩放 224x224、转 Tensor、ImageNet 归一化。"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_checkpoint(model_path, device):
    """兼容 PyTorch 2.6+，显式设置 weights_only=False。"""
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def calc_score(labels, deducts):
    """根据标签和扣分值计算最终得分（满分10分，最低0分）。"""
    total_deduct = 0

    for flag, deduct in zip(labels, deducts):
        if int(flag) == 1:
            total_deduct += deduct

    final_score = max(0, 10 - total_deduct)

    return final_score, total_deduct


def main():
    """加载模型，逐张推理 holdout 图片，对比人工标注输出评估指标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="outside_holdout_rows.csv")
    parser.add_argument("--model", default="models/outside_resnet18.pth")
    parser.add_argument("--out", default="outside_holdout_eval_result.csv")
    args = parser.parse_args()

    holdout_path = csv_input_path(args.holdout)
    model_path = Path(args.model)

    if not holdout_path.exists():
        raise FileNotFoundError(f"找不到验证数据文件: {holdout_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    df = pd.read_csv(holdout_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"验证数据中找不到列: {col}")

    if "image_path" not in df.columns:
        raise ValueError("验证数据中找不到 image_path 列")

    for col in LABEL_COLS:
        df[col] = df[col].fillna(0).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备:", device)

    checkpoint = load_checkpoint(model_path, device)

    label_names = checkpoint["label_names"]
    deducts = checkpoint["deducts"]
    thresholds = np.array(checkpoint["thresholds"])

    model = build_model(num_labels=len(label_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tf = get_transform()

    all_true = []
    all_pred = []
    all_prob = []
    result_rows = []

    for _, row in df.iterrows():
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
        raise ValueError("没有成功测试任何图片，请检查 image_path 是否正确")

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    result_df = pd.DataFrame(result_rows)
    out_path = csv_output_path(args.out)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    label_acc = (y_true == y_pred).mean()
    exact_match = np.mean([np.array_equal(t, p) for t, p in zip(y_true, y_pred)])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    mean_score_error = result_df["得分误差"].mean()

    print("\n========== 房前屋后模型验证结果 ==========")
    print("验证图片数量:", len(result_df))
    print("标签平均准确率:", round(label_acc, 4))
    print("整张图片标签完全一致率:", round(exact_match, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print("Macro Recall:", round(macro_recall, 4))
    print("平均得分误差:", round(mean_score_error, 4))
    print("详细结果已保存:", out_path)

    print("\n各标签 F1：")
    for i, name in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        print(f"label_{i} {name}: F1={f1:.4f}")


if __name__ == "__main__":
    main()



