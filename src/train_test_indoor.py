"""
室内场景模型训练与测试脚本。
从 all_labels.csv 中筛选 scene=室内 的数据，按 8:2 划分训练/测试集，
基于 ResNet18 进行多标签分类训练，每轮在测试集上评估并保存最佳模型。

用法: python train_test_indoor.py [--csv <CSV路径>] [--epochs 30] [--batch_size 8]
"""

import argparse
import copy
from pathlib import Path

try:
    from src.output_utils import csv_output_path
except ModuleNotFoundError:
    from output_utils import csv_output_path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score


LABEL_NAMES = [
    "A0_室内格局杂乱无章",
    "A1_室内家具摆放杂乱无章",
    "A2_室内生活用品杂乱无章",
    "A3_鸡鸭是否进入屋内共居",
    "A4_室内地面存在鸡鸭粪污",
    "A5_鸡跳在室内桌子上",
    "A6_室内地面垃圾乱丢现象严重",
    "A7_室内桌面沙发表面乱堆乱摆",
    "A8_室内墙面污迹不堪",
    "A9_室内其他脏乱情况",
]

LABEL_COLS = [f"label_{i}" for i in range(10)]

DEDUCTS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

THRESHOLDS = np.array([
    0.65,  # A0 室内格局杂乱无章
    0.65,  # A1 室内家具摆放杂乱无章
    0.65,  # A2 室内生活用品杂乱无章
    0.65,  # A3 鸡鸭是否进入屋内共居
    0.65,  # A4 室内地面存在鸡鸭粪污
    0.65,  # A5 鸡跳在室内桌子上
    0.60,  # A6 室内地面垃圾乱丢现象严重
    0.60,  # A7 室内桌面沙发表面乱堆乱摆
    0.60,  # A8 室内墙面污迹不堪
    0.65,  # A9 室内其他脏乱情况
])


def read_csv_safely(csv_path: Path):
    """
    自动尝试多种编码读取 CSV。
    适合 Excel / WPS 导出的中文 CSV。
    """
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    last_error = None

    for enc in encodings:
        try:
            print(f"尝试使用编码读取 CSV: {enc}")
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            print(f"读取成功，使用编码: {enc}")
            print("CSV列名:", df.columns.tolist())
            return df
        except Exception as e:
            last_error = e
            print(f"编码 {enc} 读取失败: {e}")

    raise last_error


class IndoorDataset(Dataset):
    """室内场景的自定义 Dataset，从 DataFrame 中读取图片和标签。"""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")

        labels = row[LABEL_COLS].values.astype("float32")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)


def build_model(num_labels=10, freeze_backbone=True):
    """
    构建 ResNet18 多标签分类模型。
    freeze_backbone=True 时，只训练最后一层，适合小数据集。
    """
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("已加载 ResNet18 预训练权重")
    except Exception as e:
        print("预训练权重加载失败，将使用随机初始化。原因：", e)
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)

    return model


def get_transforms():
    """返回 (train_transform, test_transform)，训练用数据增强，测试仅缩放归一化。"""
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tf, test_tf


def make_pos_weight(train_df):
    """
    处理正负样本不均衡。
    例如某个标签 1 很少，给它更高权重。
    """
    labels = train_df[LABEL_COLS].values.astype("float32")

    pos = labels.sum(axis=0)
    neg = len(labels) - pos

    weights = []

    for p, n in zip(pos, neg):
        if p < 1:
            weights.append(1.0)
        else:
            w = n / p
            w = min(max(w, 1.0), 10.0)
            weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def calc_score(pred_labels):
    """根据预测标签计算扣分和最终得分（满分10分）。"""
    total_deduct = 0

    for flag, deduct in zip(pred_labels, DEDUCTS):
        if flag == 1:
            total_deduct += deduct

    final_score = max(0, 10 - total_deduct)

    return final_score, total_deduct


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= THRESHOLDS).astype(int)

    avg_loss = total_loss / max(len(loader), 1)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, f1, precision, recall, y_true, y_prob, y_pred


def save_test_results(test_df, y_true, y_prob, y_pred, out_path):
    rows = []

    for i in range(len(test_df)):
        row = {
            "house_id": test_df.iloc[i]["house_id"],
            "image_path": test_df.iloc[i]["image_path"],
        }

        true_score, true_deduct = calc_score(y_true[i])
        pred_score, pred_deduct = calc_score(y_pred[i])

        row["人工扣分"] = true_deduct
        row["人工得分"] = true_score
        row["模型扣分"] = pred_deduct
        row["模型得分"] = pred_score

        for j in range(10):
            row[f"true_label_{j}"] = int(y_true[i][j])
            row[f"prob_label_{j}"] = round(float(y_prob[i][j]), 4)
            row[f"pred_label_{j}"] = int(y_pred[i][j])

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"测试结果已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = read_csv_safely(csv_path)

    # 清理列名，避免列名中有空格
    df.columns = [str(c).strip() for c in df.columns]

    if "scene" not in df.columns:
        raise ValueError("CSV 中找不到 scene 列，请检查 all_labels.csv 的列名")

    if "image_path" not in df.columns:
        raise ValueError("CSV 中找不到 image_path 列，请检查 all_labels.csv 的列名")

    if "house_id" not in df.columns:
        print("CSV 中没有 house_id 列，将自动创建")
        df["house_id"] = range(len(df))

    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"CSV 中找不到 {col} 列，请检查 all_labels.csv")

    df["scene"] = df["scene"].astype(str).str.strip()

    indoor_df = df[df["scene"] == "室内"].copy()

    if len(indoor_df) == 0:
        print("当前 CSV 中的 scene 唯一值：")
        print(df["scene"].unique())
        raise ValueError("CSV 里没有找到 scene == 室内 的数据")

    for col in LABEL_COLS:
        indoor_df[col] = indoor_df[col].fillna(0).astype(int)

    indoor_df["image_path"] = indoor_df["image_path"].astype(str)

    # 检查图片是否存在
    valid_rows = []

    for _, row in indoor_df.iterrows():
        image_path = Path(row["image_path"])

        if image_path.exists():
            valid_rows.append(row)
        else:
            print("图片不存在，已跳过:", row["image_path"])

    indoor_df = pd.DataFrame(valid_rows)

    print("\n室内图片数量:", len(indoor_df))

    if len(indoor_df) < 10:
        raise ValueError("室内图片太少，建议至少 30 张以上")

    print("\n室内标签正样本数量:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        print(f"{col} {name}: {int(indoor_df[col].sum())}")

    train_df, test_df = train_test_split(
        indoor_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print("\n训练集数量:", len(train_df))
    print("测试集数量:", len(test_df))

    train_tf, test_tf = get_transforms()

    train_dataset = IndoorDataset(train_df, train_tf)
    test_dataset = IndoorDataset(test_df, test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n当前设备:", device)

    model = build_model(num_labels=10, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    pos_weight = make_pos_weight(train_df).to(device)
    print("\npos_weight:", pos_weight.detach().cpu().numpy())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_loss = float("inf")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / "indoor_resnet18.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / max(len(train_loader), 1)

        test_loss, f1, precision, recall, y_true, y_prob, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"F1={f1:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss

            checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "label_names": LABEL_NAMES,
                "label_cols": LABEL_COLS,
                "deducts": DEDUCTS,
                "thresholds": THRESHOLDS.tolist()
            }

            torch.save(checkpoint, save_path)
            print("保存最佳模型:", save_path)

    print("\n训练结束")
    print("最佳测试损失:", best_loss)
    print("模型保存位置:", save_path)

    # 加载最佳模型，重新评估并保存测试结果
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, f1, precision, recall, y_true, y_prob, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    save_test_results(
        test_df.reset_index(drop=True),
        y_true,
        y_prob,
        y_pred,
        csv_output_path("indoor_test_result.csv")
    )

    print("\n最终测试结果:")
    print("F1:", round(f1, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))


if __name__ == "__main__":
    main()

