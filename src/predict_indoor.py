import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def build_model(num_labels=10):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/indoor_resnet18.pth")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

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
    print("图片:", image_path)
    print("模型:", model_path)
    print("场景: 室内")
    print("==============================\n")

    for i, name in enumerate(label_names):
        prob = probs[i]
        threshold = thresholds[i]
        is_deduct = pred_labels[i] == 1

        print(
            f"label_{i} {name} | "
            f"概率={prob:.3f} | "
            f"阈值={threshold:.2f} | "
            f"{'扣分' if is_deduct else '不扣'}"
        )

        if is_deduct:
            total_deduct += deducts[i]

    final_score = max(0, 10 - total_deduct)

    print("\n总扣分:", total_deduct)
    print(f"室内得分: {final_score}/10")


if __name__ == "__main__":
    main()