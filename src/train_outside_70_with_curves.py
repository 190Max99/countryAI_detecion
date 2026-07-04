"""
房前屋后场景模型训练脚本：带训练过程曲线版本。

功能：
1. 从 all_labels.csv 中取出前 train_num 组房前屋后数据作为训练集；
2. 剩余数据作为 holdout 验证集，并保存 outside_holdout_rows.csv；
3. 基于 ResNet18 进行多标签分类训练；
4. 每个 epoch 记录训练 Loss、验证 Loss、标签准确率、整图完全一致率、评分 MAE、评分分差；
5. 训练结束后自动保存 history CSV 和多张曲线图。

用法：
python -m src.train_outside_70_with_curves --csv data/all_labels.csv --epochs 40 --train_num 70
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

import matplotlib.pyplot as plt


SCENE_NAME = "房前屋后"
BASE_SCORE = 10

LABEL_NAMES = [
    "D0_房屋旁柴草堆码乱堆不整齐",
    "D1_房屋周身存在污水横流现象",
    "D2_房屋周身瓜果棚架破败不堪",
    "D3_房屋周身鸡鸭棚圈破败不堪脏臭",
    "D4_房屋周身其他情况",
]

LABEL_COLS = [f"label_{i}" for i in range(5)]

DEDUCTS = np.array([3, 2, 2, 2, 1], dtype=np.float32)

THRESHOLDS = np.array([
    0.55,  # D0 柴草堆码乱堆、不整齐
    0.50,  # D1 污水横流
    0.50,  # D2 瓜果棚架破败
    0.50,  # D3 鸡鸭棚圈破败、脏臭
    0.65,  # D4 其他情况
], dtype=np.float32)


# =========================
# 1. 工具函数
# =========================

def read_csv_safely(csv_path: Path):
    """尝试多种编码读取 CSV，兼容 Excel/WPS 导出的中文文件。"""
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
    """尽量按照数字顺序排序，兼容非纯数字 house_id。"""
    text = str(value).strip()

    try:
        return int(text)
    except Exception:
        return text


def set_matplotlib_chinese_font():
    """尽量设置中文字体，避免曲线图中文乱码。"""
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    plt.rcParams["font.sans-serif"] = candidate_fonts
    plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. Dataset 与 transform
# =========================

class OutsideDataset(Dataset):
    """房前屋后场景 Dataset，从 DataFrame 中读取图片和标签。"""

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


def get_train_transform():
    """训练数据增强：随机裁剪、翻转、旋转、颜色抖动 + 归一化。"""
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


def get_eval_transform():
    """验证/绘制曲线时使用固定预处理，不做随机增强。"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# 3. 模型与损失
# =========================

def build_model(num_labels=5, freeze_backbone=True):
    """
    使用 ResNet18 作为多标签分类模型。
    小数据集建议 freeze_backbone=True，只训练最后分类层。
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


def make_pos_weight(train_df):
    """处理标签不均衡。某个扣分项正样本越少，它的权重越大。"""
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


def calc_score_from_labels(labels_np):
    """
    根据标签矩阵计算每张图片得分。

    labels_np: shape = [N, num_labels]，元素为 0/1。
    """
    deduct = (labels_np * DEDUCTS).sum(axis=1)
    score = np.maximum(0, BASE_SCORE - deduct)
    return score, deduct


# =========================
# 4. 每轮评估指标
# =========================

@torch.no_grad()
def evaluate_model(model, loader, criterion, device, thresholds=THRESHOLDS):
    """
    计算当前模型在某个数据集上的指标。

    返回：
    loss：BCEWithLogitsLoss
    label_accuracy：所有标签逐项准确率
    exact_match：整张图所有标签完全一致率
    score_mae：评分平均绝对误差
    score_error_mean：平均分差，预测得分 - 实际得分
    score_exact_accuracy：最终得分完全一致率
    """
    if loader is None or len(loader.dataset) == 0:
        return {
            "loss": np.nan,
            "label_accuracy": np.nan,
            "exact_match": np.nan,
            "score_mae": np.nan,
            "score_error_mean": np.nan,
            "score_exact_accuracy": np.nan,
        }

    model.eval()

    losses = []
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        losses.append(loss.item())
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    targets_np = np.concatenate(all_targets, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    pred_np = (probs_np >= thresholds.reshape(1, -1)).astype(np.float32)

    label_accuracy = (pred_np == targets_np).mean()
    exact_match = (pred_np == targets_np).all(axis=1).mean()

    true_score, _ = calc_score_from_labels(targets_np)
    pred_score, _ = calc_score_from_labels(pred_np)

    score_error = pred_score - true_score
    score_mae = np.abs(score_error).mean()
    score_error_mean = score_error.mean()
    score_exact_accuracy = (score_error == 0).mean()

    return {
        "loss": float(np.mean(losses)),
        "label_accuracy": float(label_accuracy),
        "exact_match": float(exact_match),
        "score_mae": float(score_mae),
        "score_error_mean": float(score_error_mean),
        "score_exact_accuracy": float(score_exact_accuracy),
    }


def format_metric(value, digits=4):
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


# =========================
# 5. 绘制训练曲线
# =========================

def plot_one_curve(history_df, x_col, y_cols, labels, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))

    has_line = False

    for y_col, label in zip(y_cols, labels):
        if y_col not in history_df.columns:
            continue

        valid_df = history_df[[x_col, y_col]].dropna()

        if len(valid_df) == 0:
            continue

        plt.plot(valid_df[x_col], valid_df[y_col], marker="o", label=label)
        has_line = True

    if not has_line:
        plt.close()
        return False

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def plot_training_curves(history_df, output_dir: Path, prefix="outside"):
    """根据 history DataFrame 绘制多张训练过程曲线。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    set_matplotlib_chinese_font()

    curve_specs = [
        {
            "filename": f"{prefix}_loss_curve.png",
            "y_cols": ["train_loss", "train_eval_loss", "val_loss"],
            "labels": ["训练Loss-带数据增强", "训练Loss-固定预处理", "验证Loss"],
            "title": "训练与验证 Loss 曲线",
            "ylabel": "Loss",
        },
        {
            "filename": f"{prefix}_label_accuracy_curve.png",
            "y_cols": ["train_label_accuracy", "val_label_accuracy"],
            "labels": ["训练标签准确率", "验证标签准确率"],
            "title": "多标签逐项准确率曲线",
            "ylabel": "Label Accuracy",
        },
        {
            "filename": f"{prefix}_exact_match_curve.png",
            "y_cols": ["train_exact_match", "val_exact_match"],
            "labels": ["训练整图完全一致率", "验证整图完全一致率"],
            "title": "整图标签完全一致率曲线",
            "ylabel": "Exact Match",
        },
        {
            "filename": f"{prefix}_score_mae_curve.png",
            "y_cols": ["train_score_mae", "val_score_mae"],
            "labels": ["训练评分MAE", "验证评分MAE"],
            "title": "评分平均绝对误差 MAE 曲线",
            "ylabel": "MAE",
        },
        {
            "filename": f"{prefix}_score_error_curve.png",
            "y_cols": ["train_score_error_mean", "val_score_error_mean"],
            "labels": ["训练平均分差", "验证平均分差"],
            "title": "平均分差曲线：预测得分 - 实际得分",
            "ylabel": "Mean Score Error",
        },
        {
            "filename": f"{prefix}_score_exact_accuracy_curve.png",
            "y_cols": ["train_score_exact_accuracy", "val_score_exact_accuracy"],
            "labels": ["训练得分完全一致率", "验证得分完全一致率"],
            "title": "最终得分完全一致率曲线",
            "ylabel": "Score Exact Accuracy",
        },
    ]

    saved = []

    for spec in curve_specs:
        save_path = output_dir / spec["filename"]
        ok = plot_one_curve(
            history_df=history_df,
            x_col="epoch",
            y_cols=spec["y_cols"],
            labels=spec["labels"],
            title=spec["title"],
            ylabel=spec["ylabel"],
            save_path=save_path,
        )

        if ok:
            saved.append(save_path)
            print("已保存曲线:", save_path)

    return saved


# =========================
# 6. 主训练流程
# =========================

def main():
    """数据准备、模型构建、训练循环、保存最佳模型、保存曲线。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_num", type=int, default=70)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--output_dir", default="outputs/training_curves")
    parser.add_argument("--save_prefix", default="outside")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = read_csv_safely(csv_path)

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["house_id", "scene", "image_path"] + LABEL_COLS

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中找不到列: {col}")

    df["scene"] = df["scene"].astype(str).str.strip()

    outside_df = df[df["scene"] == SCENE_NAME].copy()

    if len(outside_df) == 0:
        print("当前 CSV 中的 scene 唯一值：")
        print(df["scene"].unique())
        raise ValueError(f"CSV 里没有找到 scene == {SCENE_NAME} 的数据")

    for col in LABEL_COLS:
        outside_df[col] = outside_df[col].fillna(0).astype(int)

    outside_df["image_path"] = outside_df["image_path"].astype(str)

    valid_rows = []

    for _, row in outside_df.iterrows():
        image_path = Path(row["image_path"])

        if image_path.exists():
            valid_rows.append(row)
        else:
            print("图片不存在，已跳过:", row["image_path"])

    outside_df = pd.DataFrame(valid_rows)

    if len(outside_df) == 0:
        raise ValueError("没有可用的房前屋后图片，请检查 image_path")

    outside_df["house_sort_key"] = outside_df["house_id"].apply(sort_house_id)
    outside_df = outside_df.sort_values(by="house_sort_key").reset_index(drop=True)

    train_df = outside_df.iloc[:args.train_num].copy()
    holdout_df = outside_df.iloc[args.train_num:].copy()

    if len(train_df) < args.train_num:
        print(f"警告：当前可用房前屋后数据只有 {len(train_df)} 张，不足 {args.train_num} 张")

    holdout_save_path = csv_output_path("outside_holdout_rows.csv")
    holdout_df.drop(columns=["house_sort_key"], errors="ignore").to_csv(
        holdout_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    train_df = train_df.drop(columns=["house_sort_key"], errors="ignore")
    holdout_df = holdout_df.drop(columns=["house_sort_key"], errors="ignore")

    print("\n========== 数据划分 ==========")
    print("房前屋后总图片数量:", len(outside_df))
    print("训练图片数量:", len(train_df))
    print("保留验证图片数量:", len(holdout_df))
    print("保留验证数据已保存:", holdout_save_path)

    print("\n训练用 house_id:")
    print(train_df["house_id"].tolist())

    print("\n房前屋后标签正样本数量:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        print(f"{col} {name}: {int(train_df[col].sum())}")

    print("\n标签检查:")
    for col, name in zip(LABEL_COLS, LABEL_NAMES):
        pos_count = int(train_df[col].sum())
        if pos_count == 0:
            print(f"警告：{col} {name} 在前{args.train_num}组训练集中没有正样本，模型很难学会这个标签")

    train_dataset = OutsideDataset(
        train_df,
        transform=get_train_transform()
    )

    train_eval_dataset = OutsideDataset(
        train_df,
        transform=get_eval_transform()
    )

    holdout_dataset = OutsideDataset(
        holdout_df,
        transform=get_eval_transform()
    ) if len(holdout_df) > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    ) if holdout_dataset is not None else None

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

    best_monitor_value = float("inf")
    best_monitor_name = "val_loss" if holdout_loader is not None else "train_loss"

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / "outside_resnet18.pth"

    history = []

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

        train_metrics = evaluate_model(
            model=model,
            loader=train_eval_loader,
            criterion=criterion,
            device=device,
            thresholds=THRESHOLDS,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=holdout_loader,
            criterion=criterion,
            device=device,
            thresholds=THRESHOLDS,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_metrics["loss"],
            "train_label_accuracy": train_metrics["label_accuracy"],
            "train_exact_match": train_metrics["exact_match"],
            "train_score_mae": train_metrics["score_mae"],
            "train_score_error_mean": train_metrics["score_error_mean"],
            "train_score_exact_accuracy": train_metrics["score_exact_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_label_accuracy": val_metrics["label_accuracy"],
            "val_exact_match": val_metrics["exact_match"],
            "val_score_mae": val_metrics["score_mae"],
            "val_score_error_mean": val_metrics["score_error_mean"],
            "val_score_exact_accuracy": val_metrics["score_exact_accuracy"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={format_metric(train_loss)}, "
            f"train_acc={format_metric(row['train_label_accuracy'])}, "
            f"train_MAE={format_metric(row['train_score_mae'])}, "
            f"val_loss={format_metric(row['val_loss'])}, "
            f"val_acc={format_metric(row['val_label_accuracy'])}, "
            f"val_MAE={format_metric(row['val_score_mae'])}"
        )

        if holdout_loader is not None and not pd.isna(row["val_loss"]):
            monitor_value = row["val_loss"]
        else:
            monitor_value = train_loss

        if monitor_value < best_monitor_value:
            best_monitor_value = monitor_value

            checkpoint = {
                "scene": SCENE_NAME,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "label_names": LABEL_NAMES,
                "label_cols": LABEL_COLS,
                "deducts": DEDUCTS.astype(int).tolist(),
                "thresholds": THRESHOLDS.tolist(),
                "train_house_ids": train_df["house_id"].tolist(),
                "train_num": len(train_df),
                "best_monitor_name": best_monitor_name,
                "best_monitor_value": best_monitor_value,
            }

            torch.save(checkpoint, save_path)
            print(f"保存当前最佳模型: {save_path}，依据 {best_monitor_name}={best_monitor_value:.4f}")

        # 每个 epoch 都保存一次 history，训练中断也能保留曲线数据
        history_df = pd.DataFrame(history)
        history_csv_path = csv_output_path(f"{args.save_prefix}_training_history.csv")
        history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")

    print("\n训练结束")
    print(f"最佳指标: {best_monitor_name} = {best_monitor_value:.4f}")
    print("模型保存位置:", save_path)

    history_df = pd.DataFrame(history)
    history_csv_path = csv_output_path(f"{args.save_prefix}_training_history.csv")
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print("训练历史已保存:", history_csv_path)

    saved_curves = plot_training_curves(
        history_df=history_df,
        output_dir=output_dir,
        prefix=args.save_prefix,
    )

    print("\n曲线输出目录:", output_dir)
    for path in saved_curves:
        print("-", path)

    print("\n下一步可以查看:")
    print(history_csv_path)
    print(output_dir)


if __name__ == "__main__":
    main()


