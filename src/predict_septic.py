import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def build_model(num_labels=3):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="要测试的化粪池图片路径")
    parser.add_argument("--model", default="models/septic_resnet18.pth", help="化粪池模型路径")
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

    label_names = checkpoint.get("label_names", [
        "C2_化粪池盖板挪开未关闭取粪口未关闭",
        "C3_化粪池粪污溢流",
        "C4_厕所周围其他情况",
    ])

    deducts = checkpoint.get("deducts", [2, 2, 1])

    thresholds = np.array(checkpoint.get("thresholds", [
        0.50,
        0.50,
        0.65,
    ]))

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
    print("场景: 化粪池")
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

    print("\n========== 评分结果 ==========")
    print("化粪池部分总扣分:", total_deduct)
    print("说明：化粪池不是单独10分，需与厕所图片扣分合并计算。")
    print("厕所及化粪池得分 = 10 - 厕所扣分 - 化粪池扣分")


if __name__ == "__main__":
    main()