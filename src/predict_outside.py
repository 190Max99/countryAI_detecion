"""
房前屋后场景单张图片预测脚本。
加载 outside_resnet18.pth 模型，对单张图片进行多标签分类，
输出每个标签的概率、阈值对比及最终扣分和得分（满分10分）。

用法: python predict_outside.py --image <图片路径> [--model <模型路径>]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def build_model(num_labels=5):
    """构建 ResNet18 多标签分类模型（房前屋后5个标签）。"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
    return model


def get_transform():
    """图像预处理：缩放到 224x224、转 Tensor、ImageNet 归一化。"""
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


def main():
    """解析参数、加载模型、单张图片推理并输出评分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="要测试的房前屋后图片路径")
    parser.add_argument("--model", default="models/outside_resnet18.pth", help="模型路径")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")

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

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pred_labels = (probs >= thresholds).astype(int)

    total_deduct = 0

    print("\n==============================")
    print("场景: 房前屋后")
    print("图片:", image_path)
    print("模型:", model_path)
    print("==============================\n")

    for i, name in enumerate(label_names):
        prob = probs[i]
        threshold = thresholds[i]
        is_deduct = pred_labels[i] == 1

        if is_deduct:
            total_deduct += deducts[i]

        print(
            f"label_{i} {name} | "
            f"概率={prob:.3f} | "
            f"阈值={threshold:.2f} | "
            f"{'扣分' if is_deduct else '不扣'} | "
            f"扣分值={deducts[i]}"
        )

    final_score = max(0, 10 - total_deduct)

    print("\n========== 评分结果 ==========")
    print("总扣分:", total_deduct)
    print(f"房前屋后得分: {final_score}/10")


if __name__ == "__main__":
    main()