"""
室内模型全量验证脚本。
读取 all_labels.csv 中 scene=室内 的所有数据，用 indoor_resnet18.pth 逐张推理，
对比人工标注与模型预测，输出：标签准确率、F1/Precision/Recall、得分误差等。

用法: python eval_indoor_current.py [--csv <CSV路径>] [--model <模型路径>] [--out <输出CSV>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.metrics import f1_score, precision_score, recall_score


# 室内场景有 10 个标签列
LABEL_COLS = [f"label_{i}" for i in range(10)]


def read_csv_safely(csv_path: Path):
    """尝试多种编码读取 CSV，兼容 Excel/WPS 导出的中文文件。"""
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


def build_model(num_labels=10):
    """构建 ResNet18 多标签分类模型，替换全连接层。"""
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
            total_deduct += int(deduct)

    final_score = max(0, 10 - total_deduct)

    return final_score, total_deduct


def main():
    """加载模型，逐张推理室内图片，对比人工标注输出评估指标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--model", default="models/indoor_resnet18.pth")
    parser.add_argument("--out", default="indoor_eval_current_result.csv")
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

    # 筛选室内场景数据
    indoor_df = df[df["scene"] == "室内"].copy()

    if len(indoor_df) == 0:
        print("当前 CSV 中 scene 的取值：")
        print(df["scene"].unique())
        raise ValueError("没有找到 scene == 室内 的数据")

    for col in LABEL_COLS:
        indoor_df[col] = indoor_df[col].fillna(0).astype(int)

    indoor_df["image_path"] = indoor_df["image_path"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n当前设备:", device)

    # 加载 checkpoint：包含权重、标签名、扣分值、阈值
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

    for _, row in indoor_df.iterrows():
        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            print("图片不存在，跳过:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = tf(image).unsqueeze(0).to(device)

        # 推理 + sigmoid 得到概率
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        true_labels = row[LABEL_COLS].values.astype(int)
        pred_labels = (probs >= thresholds).astype(int)  # 阈值判断

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

    # 整体指标计算
    label_acc = (y_true == y_pred).mean()
    exact_match = np.mean([np.array_equal(t, p) for t, p in zip(y_true, y_pred)])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    mean_score_error = result_df["得分误差"].mean()

    print("\n========== 室内模型重新测评结果 ==========")
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