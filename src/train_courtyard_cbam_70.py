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


SCENE_NAME = "庭院"

LABEL_NAMES = [
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
]

LABEL_COLS = [f"label_{i}" for i in range(12)]

DEDUCTS = [3, 3, 2, 2, 2, 5, 1, 2, 2, 5, 2, 1]

THRESHOLDS = np.array([
    0.55,
    0.55,
    0.60,
    0.45,
    0.45,
    0.50,
    0.55,
    0.55,
    0.55,
    0.55,
    0.50,
    0.65,
])


# =========================
# 1. CBAM 注意力模块
# =========================

class ChannelAttention(nn.Module):
    """
    通道注意力模块。

    作用：
    让模型判断哪些特征通道更重要。
    例如庭院图片中，与垃圾、污水、柴草、棚架相关的特征通道应该被增强。
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        hidden_channels = max(in_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块。

    作用：
    让模型判断图片中哪些区域更重要。
    例如庭院图片中的垃圾区域、污水区域、柴草堆区域、破败棚架区域。
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))

        return x * attention


class CBAM(nn.Module):
    """
    CBAM = Channel Attention + Spatial Attention

    先做通道注意力，再做空间注意力。
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x


class ResNet18CBAM(nn.Module):
    """
    ResNet18 + CBAM。

    改动位置：
    ResNet18 的 layer4 后面加入 CBAM。

    原始流程：
    image -> ResNet18 -> avgpool -> fc

    修改后：
    image -> ResNet18 -> CBAM -> avgpool -> fc
    """

    def __init__(self, num_labels=12, freeze_backbone=True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
            print("已加载 ResNet18 预训练权重")
        except Exception as e:
            print("预训练权重加载失败，将使用随机初始化。原因：", e)
            base_model = models.resnet18(weights=None)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.cbam = CBAM(in_channels=512)

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_labels)

        if freeze_backbone:
            self.freeze_resnet_backbone()

    def freeze_resnet_backbone(self):
        """
        只冻结 ResNet18 主干特征提取部分。
        CBAM 和 fc 分类头保持可训练。
        """

        backbone_modules = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = False

        for param in self.cbam.parameters():
            param.requires_grad = True

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.cbam(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# =========================
# 2. 数据读取
# =========================

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


class CourtyardDataset(Dataset):
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


# =========================
# 3. 模型构建
# =========================

def build_model(num_labels=12, freeze_backbone=True):
    """
    构建 ResNet18 + CBAM 模型。
    """
    model = ResNet18CBAM(
        num_labels=num_labels,
        freeze_backbone=freeze_backbone,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n========== 模型结构 ==========")
    print("模型类型: ResNet18 + CBAM")
    print("总参数量:", total_params)
    print("可训练参数量:", trainable_params)

    if freeze_backbone:
        print("当前设置: 冻结 ResNet18 主干，只训练 CBAM 和 fc 分类头")
    else:
        print("当前设置: 训练整个 ResNet18 + CBAM 模型")

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


def calc_score(labels):
    total_deduct = 0

    for flag, deduct in zip(labels, DEDUCTS):
        if int(flag) == 1:
            total_deduct += deduct

    final_score = max(0, 30 - total_deduct)

    return final_score, total_deduct


# =========================
# 4. 主训练流程
# =========================

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

    courtyard_df = df[df["scene"] == SCENE_NAME].copy()

    if len(courtyard_df) == 0:
        print("当前 CSV 中的 scene 唯一值：")
        print(df["scene"].unique())
        raise ValueError(f"CSV 里没有找到 scene == {SCENE_NAME} 的数据")

    for col in LABEL_COLS:
        courtyard_df[col] = courtyard_df[col].fillna(0).astype(int)

    courtyard_df["image_path"] = courtyard_df["image_path"].astype(str)

    valid_rows = []

    for _, row in courtyard_df.iterrows():
        image_path = Path(row["image_path"])

        if image_path.exists():
            valid_rows.append(row)
        else:
            print("图片不存在，已跳过:", row["image_path"])

    courtyard_df = pd.DataFrame(valid_rows)

    if len(courtyard_df) == 0:
        raise ValueError("没有可用的庭院图片，请检查 image_path")

    courtyard_df["house_sort_key"] = courtyard_df["house_id"].apply(sort_house_id)
    courtyard_df = courtyard_df.sort_values(by="house_sort_key").reset_index(drop=True)

    train_df = courtyard_df.iloc[:args.train_num].copy()
    holdout_df = courtyard_df.iloc[args.train_num:].copy()

    if len(train_df) < args.train_num:
        print(f"警告：当前可用庭院数据只有 {len(train_df)} 张，不足 {args.train_num} 张")

    holdout_save_path = csv_output_path("courtyard_cbam_holdout_rows.csv")
    holdout_df.drop(columns=["house_sort_key"], errors="ignore").to_csv(
        holdout_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    train_df = train_df.drop(columns=["house_sort_key"], errors="ignore")

    print("\n========== 数据划分 ==========")
    print("庭院总图片数量:", len(courtyard_df))
    print("训练图片数量:", len(train_df))
    print("保留验证图片数量:", len(holdout_df))
    print("保留验证数据已保存:", holdout_save_path)

    print("\n训练用 house_id:")
    print(train_df["house_id"].tolist())

    print("\n庭院标签正样本数量:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        print(f"{col} {name}: {int(train_df[col].sum())}")

    print("\n标签检查:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        pos_count = int(train_df[col].sum())
        if pos_count == 0:
            print(f"警告：{col} {name} 在前 {len(train_df)} 组训练集中没有正样本，模型很难学会这个标签")

    train_dataset = CourtyardDataset(
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

    save_path = model_dir / "courtyard_resnet18_cbam.pth"

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
                "model_type": "resnet18_cbam",
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "label_names": LABEL_NAMES,
                "label_cols": LABEL_COLS,
                "deducts": DEDUCTS,
                "thresholds": THRESHOLDS.tolist(),
                "train_house_ids": train_df["house_id"].tolist(),
                "train_num": len(train_df),
                "freeze_backbone": args.freeze_backbone,
            }

            torch.save(checkpoint, save_path)
            print("保存当前最佳 CBAM 模型:", save_path)

    print("\n训练结束")
    print("最佳训练损失:", best_train_loss)
    print("模型保存位置:", save_path)

    print("\n下一步可以用剩余数据验证：")
    print(holdout_save_path)


if __name__ == "__main__":
    main()



