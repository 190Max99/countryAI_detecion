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


SCENE_NAME = "厕所"

LABEL_NAMES = [
    "C0_厕屋脏乱",
    "C1_厕屋功能配备不齐全",
]

LABEL_COLS = [f"label_{i}" for i in range(2)]

DEDUCTS = [2, 3]

THRESHOLDS = np.array([
    0.50,  # C0 厕屋脏乱
    0.50,  # C1 厕屋功能配备不齐全
])


def read_csv_safely(csv_path: Path):
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


def sort_house_id(value):
    text = str(value).strip()

    try:
        return int(text)
    except Exception:
        return text


class ToiletDataset(Dataset):
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


def build_model(num_labels=2, freeze_backbone=True):
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


def get_train_transform():
    return transforms.Compose([
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


def make_pos_weight(train_df):
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


def calc_deduct(labels):
    total_deduct = 0

    for flag, deduct in zip(labels, DEDUCTS):
        if int(flag) == 1:
            total_deduct += deduct

    return total_deduct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_num", type=int, default=70)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = read_csv_safely(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["house_id", "scene", "image_path"] + LABEL_COLS

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中找不到列: {col}")

    df["scene"] = df["scene"].astype(str).str.strip()

    toilet_df = df[df["scene"] == SCENE_NAME].copy()

    if len(toilet_df) == 0:
        print("当前 CSV 中的 scene 唯一值：")
        print(df["scene"].unique())
        raise ValueError(f"CSV 里没有找到 scene == {SCENE_NAME} 的数据")

    for col in LABEL_COLS:
        toilet_df[col] = toilet_df[col].fillna(0).astype(int)

    toilet_df["image_path"] = toilet_df["image_path"].astype(str)

    valid_rows = []

    for _, row in toilet_df.iterrows():
        image_path = Path(row["image_path"])

        if image_path.exists():
            valid_rows.append(row)
        else:
            print("图片不存在，已跳过:", row["image_path"])

    toilet_df = pd.DataFrame(valid_rows)

    if len(toilet_df) == 0:
        raise ValueError("没有可用的厕所图片，请检查 image_path")

    toilet_df["house_sort_key"] = toilet_df["house_id"].apply(sort_house_id)
    toilet_df = toilet_df.sort_values(by="house_sort_key").reset_index(drop=True)

    train_df = toilet_df.iloc[:args.train_num].copy()
    holdout_df = toilet_df.iloc[args.train_num:].copy()

    if len(train_df) < args.train_num:
        print(f"警告：当前可用厕所数据只有 {len(train_df)} 张，不足 {args.train_num} 张")

    holdout_save_path = csv_output_path("toilet_holdout_rows.csv")
    holdout_df.drop(columns=["house_sort_key"], errors="ignore").to_csv(
        holdout_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    train_df = train_df.drop(columns=["house_sort_key"], errors="ignore")

    print("\n========== 数据划分 ==========")
    print("厕所总图片数量:", len(toilet_df))
    print("训练图片数量:", len(train_df))
    print("保留验证图片数量:", len(holdout_df))
    print("保留验证数据已保存:", holdout_save_path)

    print("\n训练用 house_id:")
    print(train_df["house_id"].tolist())

    print("\n厕所标签正样本数量:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        print(f"{col} {name}: {int(train_df[col].sum())}")

    print("\n标签检查:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        pos_count = int(train_df[col].sum())
        if pos_count == 0:
            print(f"警告：{col} {name} 在前 {len(train_df)} 组训练集中没有正样本，模型很难学会这个标签")

    train_dataset = ToiletDataset(
        train_df,
        transform=get_train_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n当前设备:", device)

    model = build_model(
        num_labels=len(LABEL_COLS),
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    pos_weight = make_pos_weight(train_df).to(device)
    print("\npos_weight:", pos_weight.detach().cpu().numpy())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_train_loss = float("inf")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / "toilet_resnet18.pth"

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

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            checkpoint = {
                "scene": SCENE_NAME,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "label_names": LABEL_NAMES,
                "label_cols": LABEL_COLS,
                "deducts": DEDUCTS,
                "thresholds": THRESHOLDS.tolist(),
                "train_house_ids": train_df["house_id"].tolist(),
                "train_num": len(train_df)
            }

            torch.save(checkpoint, save_path)
            print("保存当前最佳模型:", save_path)

    print("\n训练结束")
    print("最佳训练损失:", best_train_loss)
    print("模型保存位置:", save_path)

    print("\n下一步可以用剩余数据验证：")
    print(holdout_save_path)

    print("\n说明：厕所模型只计算厕所部分扣分。")
    print("最终厕所及化粪池得分 = 10 - 厕所扣分 - 化粪池扣分")


if __name__ == "__main__":
    main()



