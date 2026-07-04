import threading
import re
import os
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    from src.output_utils import csv_output_path
except ModuleNotFoundError:
    from output_utils import csv_output_path


# =========================
# 1. 场景配置
# =========================

SCENE_CONFIGS = {
    "室内": {
        "aliases": ["室内"],
        "model_path": "models/indoor_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(10)],
        "label_names": [
            "A0_室内格局杂乱无章",
            "A1_室内家具摆放杂乱无章",
            "A2_室内生活用品杂乱无章",
            "A3_鸡鸭进入屋内共居",
            "A4_室内地面存在鸡鸭粪污",
            "A5_鸡跳在室内桌子上",
            "A6_室内地面垃圾乱丢现象严重",
            "A7_室内桌面沙发表面乱堆乱摆",
            "A8_室内墙面污迹不堪",
            "A9_室内其他脏乱情况",
        ],
        "deducts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "thresholds": [0.55, 0.55, 0.55, 0.45, 0.45, 0.45, 0.50, 0.50, 0.50, 0.65],
    },

    "庭院": {
        "aliases": ["庭院"],
        "model_path": "models/courtyard_resnet18_cbam.pth",
        "base_score": 30,
        "label_cols": [f"label_{i}" for i in range(12)],
        "label_names": [
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
        ],
        "deducts": [3, 3, 2, 2, 2, 5, 1, 2, 2, 5, 2, 1],
        "thresholds": [0.55, 0.55, 0.60, 0.45, 0.45, 0.50, 0.55, 0.55, 0.55, 0.55, 0.50, 0.65],
    },

    "厕所": {
        "aliases": ["厕所", "厕屋"],
        "model_path": "models/toilet_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(2)],
        "label_names": [
            "C0_厕屋脏乱",
            "C1_厕屋功能配备不齐全",
        ],
        "deducts": [2, 3],
        "thresholds": [0.50, 0.50],
    },

    "化粪池": {
        "aliases": ["化粪池"],
        "model_path": "models/septic_resnet18.pth",
        "base_score": None,
        "label_cols": [f"label_{i}" for i in range(3)],
        "label_names": [
            "C2_化粪池盖板挪开未关闭取粪口未关闭",
            "C3_化粪池粪污溢流",
            "C4_厕所周围其他情况",
        ],
        "deducts": [2, 2, 1],
        "thresholds": [0.50, 0.50, 0.65],
    },

    "房前屋后": {
        "aliases": ["房前屋后", "房屋前后", "房前屋后及2侧", "房前屋后及两侧", "屋后", "两侧", "2侧"],
        "model_path": "models/outside_resnet18.pth",
        "base_score": 10,
        "label_cols": [f"label_{i}" for i in range(5)],
        "label_names": [
            "D0_房屋旁柴草堆码乱堆不整齐",
            "D1_房屋周身存在污水横流现象",
            "D2_房屋周身瓜果棚架破败不堪",
            "D3_房屋周身鸡鸭棚圈破败不堪脏臭",
            "D4_房屋周身其他情况",
        ],
        "deducts": [3, 2, 2, 2, 1],
        "thresholds": [0.55, 0.50, 0.50, 0.50, 0.65],
    },
}


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
MODEL_CACHE = {}


# =========================
# 2. CBAM 模型结构
# =========================

class ChannelAttention(nn.Module):
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
    def __init__(self, num_labels):
        super().__init__()

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
# 3. CSV 与人工分数计算
# =========================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("\t", "")
    text = text.replace(" ", "")
    text = text.replace("\u3000", "")
    text = text.strip()

    return text


def read_csv_safely(csv_path: Path):
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    seps = [None, "\t", ","]

    best_df = None
    best_col_count = 0

    for enc in encodings:
        for sep in seps:
            try:
                if sep is None:
                    df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(csv_path, encoding=enc, sep=sep)

                df.columns = [normalize_text(c) for c in df.columns]

                if len(df.columns) > best_col_count:
                    best_df = df
                    best_col_count = len(df.columns)

                if len(df.columns) >= 3:
                    return df

            except Exception:
                pass

    if best_df is not None:
        return best_df

    raise RuntimeError(f"无法读取CSV文件: {csv_path}")


def find_label_csv(folder: Path):
    preferred = folder / f"{folder.name}.csv"

    if preferred.exists():
        return preferred

    csv_files = sorted(folder.glob("*.csv"))

    bad_keywords = ["result", "detail", "score", "预测", "结果", "simple", "ui"]

    valid_files = []

    for file in csv_files:
        name = file.name.lower()

        if any(k.lower() in name for k in bad_keywords):
            continue

        valid_files.append(file)

    if len(valid_files) > 0:
        return valid_files[0]

    if len(csv_files) > 0:
        return csv_files[0]

    return None


def prepare_label_df(label_csv: Path):
    df = read_csv_safely(label_csv)

    if "scene" in df.columns:
        scene_col = "scene"
    elif "序号" in df.columns:
        scene_col = "序号"
    elif "类别" in df.columns:
        scene_col = "类别"
    else:
        scene_col = df.columns[0]

    df["scene_norm"] = df[scene_col].apply(normalize_text)

    return df


def find_true_row(label_df: pd.DataFrame, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    rows = label_df[label_df["scene_norm"].isin(aliases)]

    if len(rows) > 0:
        return rows.iloc[0]

    for _, row in label_df.iterrows():
        scene_text = str(row["scene_norm"])

        for alias in aliases:
            if alias in scene_text:
                return row

    return None


def get_label_value(row, col):
    if row is None:
        return 0

    if col not in row.index:
        return 0

    value = row[col]

    if pd.isna(value):
        return 0

    try:
        return int(float(value))
    except Exception:
        return 0


def calc_deduct_from_row(row, scene_key):
    cfg = SCENE_CONFIGS[scene_key]

    total_deduct = 0

    for col, deduct in zip(cfg["label_cols"], cfg["deducts"]):
        flag = get_label_value(row, col)

        if flag == 1:
            total_deduct += int(deduct)

    return total_deduct


def calc_score(base_score, deduct):
    if pd.isna(deduct):
        return np.nan

    return max(0, base_score - deduct)


def calc_score_error(true_score, pred_score):
    if pd.isna(true_score) or pd.isna(pred_score):
        return np.nan

    return pred_score - true_score


# =========================
# 4. 图片、模型预测
# =========================

def find_image_in_folder(folder: Path, scene_key: str):
    aliases = SCENE_CONFIGS[scene_key]["aliases"]

    candidates = []

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        filename = normalize_text(file.name)

        for alias in aliases:
            if alias in filename:
                candidates.append(file)
                break

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: p.name)

    return candidates[0]


def build_plain_resnet18(num_labels):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)

    return model


def build_model(num_labels, use_cbam=False):
    if use_cbam:
        return ResNet18CBAM(num_labels=num_labels)

    return build_plain_resnet18(num_labels=num_labels)


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


def detect_cbam_model(checkpoint):
    if checkpoint.get("model_type", "") == "resnet18_cbam":
        return True

    state_dict = checkpoint.get("model_state_dict", {})

    for key in state_dict.keys():
        if key.startswith("cbam."):
            return True

    return False


def load_model_for_scene(scene_key, device):
    cache_key = (scene_key, str(device))

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    cfg = SCENE_CONFIGS[scene_key]
    model_path = Path(cfg["model_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"{scene_key} 模型不存在: {model_path}")

    checkpoint = load_checkpoint(model_path, device)

    deducts = checkpoint.get("deducts", cfg["deducts"])
    thresholds = np.array(checkpoint.get("thresholds", cfg["thresholds"]), dtype=float)
    label_names = checkpoint.get("label_names", cfg["label_names"])

    use_cbam = detect_cbam_model(checkpoint)

    model = build_model(
        num_labels=len(label_names),
        use_cbam=use_cbam
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    MODEL_CACHE[cache_key] = {
        "model": model,
        "deducts": deducts,
        "thresholds": thresholds,
        "label_names": label_names,
        "use_cbam": use_cbam,
    }

    return MODEL_CACHE[cache_key]


def predict_scene(scene_key, image_path, device):
    model_pack = load_model_for_scene(scene_key, device)

    model = model_pack["model"]
    deducts = model_pack["deducts"]
    thresholds = model_pack["thresholds"]
    label_names = model_pack["label_names"]

    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    pred_labels = (probs >= thresholds).astype(int)

    total_deduct = 0

    for flag, deduct in zip(pred_labels, deducts):
        if int(flag) == 1:
            total_deduct += int(deduct)

    return {
        "pred_deduct": total_deduct,
        "pred_labels": pred_labels,
        "probs": probs,
        "thresholds": thresholds,
        "deducts": deducts,
        "label_names": label_names,
    }


# =========================
# 5. 图片文字标注
# =========================

def get_chinese_font(size=24):
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def text_width(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except Exception:
        return len(text) * 12


def wrap_text(draw, text, font, max_width):
    if text_width(draw, text, font) <= max_width:
        return [text]

    lines = []
    current = ""

    for ch in text:
        test = current + ch

        if text_width(draw, test, font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = ch

    if current:
        lines.append(current)

    return lines


def annotate_image(scene_key, image_path, pred_info, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw_tmp = ImageDraw.Draw(image)

    title_size = max(24, min(38, image.width // 35))
    text_size = max(18, min(28, image.width // 45))

    font_title = get_chinese_font(title_size)
    font_text = get_chinese_font(text_size)

    label_names = pred_info["label_names"]
    pred_labels = pred_info["pred_labels"]
    probs = pred_info["probs"]
    thresholds = pred_info["thresholds"]
    deducts = pred_info["deducts"]
    pred_deduct = pred_info["pred_deduct"]

    raw_lines = [
        f"场景：{scene_key}",
        f"模型扣分：{pred_deduct}"
    ]

    has_deduct = False

    for i, flag in enumerate(pred_labels):
        if int(flag) == 1:
            has_deduct = True
            raw_lines.append(
                f"{label_names[i]}  概率:{probs[i]:.3f}  阈值:{thresholds[i]:.2f}  -{deducts[i]}分"
            )

    if not has_deduct:
        raw_lines.append("预测结果：无扣分项")

    padding = 16
    left = 20
    top = 20
    box_width = max(300, image.width - 40)
    max_text_width = box_width - padding * 2

    lines = []

    for idx, line in enumerate(raw_lines):
        font = font_title if idx == 0 else font_text
        wrapped = wrap_text(draw_tmp, line, font, max_text_width)

        for w in wrapped:
            lines.append((w, font, idx == 0))

    line_heights = []

    for line, font, _ in lines:
        try:
            bbox = draw_tmp.textbbox((0, 0), line, font=font)
            line_heights.append((bbox[3] - bbox[1]) + 10)
        except Exception:
            line_heights.append(text_size + 12)

    box_height = padding * 2 + sum(line_heights)
    box_height = min(box_height, image.height - 40)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    overlay_draw.rectangle(
        [left, top, left + box_width, top + box_height],
        fill=(255, 255, 255, 215),
        outline=(220, 0, 0, 255),
        width=3
    )

    image_rgba = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image_rgba)

    x = left + padding
    y = top + padding
    max_y = top + box_height - padding

    for (line, font, is_title), h in zip(lines, line_heights):
        if y + h > max_y:
            draw.text((x, y), "……", fill=(0, 0, 0, 255), font=font_text)
            break

        fill = (180, 0, 0, 255) if is_title else (0, 0, 0, 255)
        draw.text((x, y), line, fill=fill, font=font)
        y += h

    save_path = save_dir / f"{image_path.stem}_标注.jpg"
    image_rgba.convert("RGB").save(save_path, quality=95)

    return save_path


# =========================
# 6. Grad-CAM 热力图
# =========================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.save_activations
        )

        try:
            self.backward_handle = self.target_layer.register_full_backward_hook(
                self.save_gradients
            )
        except Exception:
            self.backward_handle = self.target_layer.register_backward_hook(
                self.save_gradients
            )

    def save_activations(self, module, inputs, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, image_tensor, target_index):
        self.model.zero_grad()

        logits = self.model(image_tensor)

        target_score = logits[0, target_index]
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM 获取特征或梯度失败")

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=False)
        cam = torch.relu(cam)

        cam = cam[0]

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()


def get_target_layer_for_gradcam(model):
    if hasattr(model, "cbam"):
        return model.cbam

    if hasattr(model, "layer4"):
        return model.layer4

    raise RuntimeError("当前模型中找不到可用于 Grad-CAM 的目标层")


def safe_filename(text):
    text = str(text)
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = text.replace(" ", "")
    text = text.replace("\t", "")
    text = text[:60]
    return text


def make_heatmap_overlay(original_image, cam_np, title_text, save_path):
    original_image = original_image.convert("RGB")

    width, height = original_image.size

    cam_img = Image.fromarray(np.uint8(cam_np * 255))
    cam_img = cam_img.resize((width, height), Image.BILINEAR)

    cam_arr = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = np.zeros((height, width, 4), dtype=np.uint8)

    heatmap[:, :, 0] = 255
    heatmap[:, :, 1] = np.uint8(120 * cam_arr)
    heatmap[:, :, 2] = 0
    heatmap[:, :, 3] = np.uint8(170 * cam_arr)

    heatmap_img = Image.fromarray(heatmap, mode="RGBA")

    result = Image.alpha_composite(
        original_image.convert("RGBA"),
        heatmap_img
    )

    panel_h = max(80, height // 10)

    panel = Image.new("RGBA", (width, panel_h), (255, 255, 255, 215))
    result.alpha_composite(panel, (0, 0))

    draw = ImageDraw.Draw(result)

    font_title = get_chinese_font(size=max(22, width // 45))
    font_text = get_chinese_font(size=max(18, width // 58))

    draw.text(
        (20, 10),
        "Grad-CAM 热力图",
        fill=(180, 0, 0, 255),
        font=font_title
    )

    draw.text(
        (20, 45),
        title_text,
        fill=(0, 0, 0, 255),
        font=font_text
    )

    result.convert("RGB").save(save_path, quality=95)


def generate_gradcam_images(scene_key, image_path, pred_info, device, save_dir: Path, max_maps=3):
    save_dir.mkdir(parents=True, exist_ok=True)

    model_pack = load_model_for_scene(scene_key, device)
    model = model_pack["model"]
    model.eval()

    label_names = pred_info["label_names"]
    pred_labels = pred_info["pred_labels"]
    probs = pred_info["probs"]

    original_image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(original_image).unsqueeze(0).to(device)

    positive_indices = []

    for i, flag in enumerate(pred_labels):
        if int(flag) == 1:
            positive_indices.append(i)

    if len(positive_indices) == 0:
        indices = [int(np.argmax(probs))]
    else:
        indices = sorted(
            positive_indices,
            key=lambda i: probs[i],
            reverse=True
        )[:max_maps]

    target_layer = get_target_layer_for_gradcam(model)

    saved_paths = []

    for idx in indices:
        gradcam = GradCAM(model, target_layer)

        try:
            cam_np = gradcam.generate(image_tensor, target_index=idx)
        finally:
            gradcam.remove_hooks()

        label_name = label_names[idx]
        prob = probs[idx]

        clean_label = safe_filename(label_name)

        save_name = f"{image_path.stem}_{clean_label}_热力图.jpg"
        save_path = save_dir / save_name

        title_text = f"{scene_key} | {label_name} | 概率:{prob:.3f}"

        make_heatmap_overlay(
            original_image=original_image,
            cam_np=cam_np,
            title_text=title_text,
            save_path=save_path
        )

        saved_paths.append(save_path)

    return saved_paths


# =========================
# 7. 核心评分函数
# =========================

def score_folder(folder: Path, log_fn=None):
    def log(message):
        if log_fn is not None:
            log_fn(message)

    label_csv = find_label_csv(folder)

    if label_csv is None:
        raise FileNotFoundError(f"文件夹中没有找到人工标签CSV: {folder}")

    label_df = prepare_label_df(label_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene_deducts = {}

    annotated_dir = folder / "annotated"
    gradcam_dir = folder / "gradcam"

    annotated_paths = []
    gradcam_paths_all = []

    for scene_key in ["室内", "庭院", "厕所", "化粪池", "房前屋后"]:
        true_row = find_true_row(label_df, scene_key)
        image_path = find_image_in_folder(folder, scene_key)

        true_deduct = np.nan if true_row is None else calc_deduct_from_row(true_row, scene_key)

        if image_path is None:
            pred_deduct = np.nan
            annotated_path = ""
            log(f"{scene_key}：未找到图片")
        else:
            pred_info = predict_scene(scene_key, image_path, device)
            pred_deduct = pred_info["pred_deduct"]

            annotated_path = annotate_image(
                scene_key=scene_key,
                image_path=image_path,
                pred_info=pred_info,
                save_dir=annotated_dir
            )

            annotated_paths.append(annotated_path)
            log(f"{scene_key}：已生成标注图 {annotated_path}")

            gradcam_paths = generate_gradcam_images(
                scene_key=scene_key,
                image_path=image_path,
                pred_info=pred_info,
                device=device,
                save_dir=gradcam_dir,
                max_maps=3
            )

            for heatmap_path in gradcam_paths:
                gradcam_paths_all.append(heatmap_path)
                log(f"{scene_key}：已生成热力图 {heatmap_path}")

        scene_deducts[scene_key] = {
            "true_deduct": true_deduct,
            "pred_deduct": pred_deduct,
            "image_path": str(image_path) if image_path is not None else "",
            "annotated_path": str(annotated_path) if image_path is not None else "",
        }

    rows = []

    def add_row(item_name, base_score, true_deduct, pred_deduct):
        true_score = calc_score(base_score, true_deduct)
        pred_score = calc_score(base_score, pred_deduct)

        rows.append({
            "项目": item_name,
            "满分": base_score,
            "实际扣分": true_deduct,
            "预测扣分": pred_deduct,
            "实际得分": true_score,
            "预测得分": pred_score,
            "分差": calc_score_error(true_score, pred_score),
        })

    add_row(
        "室内",
        10,
        scene_deducts["室内"]["true_deduct"],
        scene_deducts["室内"]["pred_deduct"]
    )

    add_row(
        "庭院",
        30,
        scene_deducts["庭院"]["true_deduct"],
        scene_deducts["庭院"]["pred_deduct"]
    )

    toilet_true = scene_deducts["厕所"]["true_deduct"]
    septic_true = scene_deducts["化粪池"]["true_deduct"]

    toilet_pred = scene_deducts["厕所"]["pred_deduct"]
    septic_pred = scene_deducts["化粪池"]["pred_deduct"]

    toilet_septic_true = np.nan if pd.isna(toilet_true) or pd.isna(septic_true) else toilet_true + septic_true
    toilet_septic_pred = np.nan if pd.isna(toilet_pred) or pd.isna(septic_pred) else toilet_pred + septic_pred

    add_row(
        "厕所及化粪池",
        10,
        toilet_septic_true,
        toilet_septic_pred
    )

    add_row(
        "房前屋后",
        10,
        scene_deducts["房前屋后"]["true_deduct"],
        scene_deducts["房前屋后"]["pred_deduct"]
    )

    result_df = pd.DataFrame(rows)

    true_scores = result_df["实际得分"].tolist()
    pred_scores = result_df["预测得分"].tolist()

    total_true_score = np.nan if any(pd.isna(x) for x in true_scores) else sum(true_scores)
    total_pred_score = np.nan if any(pd.isna(x) for x in pred_scores) else sum(pred_scores)

    result_df.loc[len(result_df)] = {
        "项目": "总分",
        "满分": 60,
        "实际扣分": 60 - total_true_score if not pd.isna(total_true_score) else np.nan,
        "预测扣分": 60 - total_pred_score if not pd.isna(total_pred_score) else np.nan,
        "实际得分": total_true_score,
        "预测得分": total_pred_score,
        "分差": calc_score_error(total_true_score, total_pred_score),
    }

    save_path = csv_output_path(f"ui_score_result_{folder.name}.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return result_df, label_csv, save_path, annotated_dir, gradcam_dir, annotated_paths, gradcam_paths_all, device


# =========================
# 8. UI 界面
# =========================

class ScoreApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI积分制现场照片评分系统")
        self.root.geometry("1100x660")

        self.selected_folder = None
        self.annotated_dir = None
        self.gradcam_dir = None

        title = tk.Label(
            root,
            text="AI积分制现场照片评分系统",
            font=("Microsoft YaHei", 18, "bold")
        )

        title.pack(pady=12)

        top_frame = tk.Frame(root)
        top_frame.pack(fill="x", padx=20)

        self.folder_label = tk.Label(
            top_frame,
            text="当前未选择文件夹",
            anchor="w",
            font=("Microsoft YaHei", 10)
        )

        self.folder_label.pack(side="left", fill="x", expand=True)

        choose_btn = tk.Button(
            top_frame,
            text="选择农户文件夹",
            command=self.choose_folder,
            width=16,
            font=("Microsoft YaHei", 10)
        )

        choose_btn.pack(side="right", padx=5)

        run_btn = tk.Button(
            top_frame,
            text="开始评分",
            command=self.run_score_thread,
            width=12,
            font=("Microsoft YaHei", 10)
        )

        run_btn.pack(side="right", padx=5)

        open_annotated_btn = tk.Button(
            top_frame,
            text="打开标注图",
            command=self.open_annotated_folder,
            width=12,
            font=("Microsoft YaHei", 10)
        )

        open_annotated_btn.pack(side="right", padx=5)

        open_gradcam_btn = tk.Button(
            top_frame,
            text="打开热力图",
            command=self.open_gradcam_folder,
            width=12,
            font=("Microsoft YaHei", 10)
        )

        open_gradcam_btn.pack(side="right", padx=5)

        columns = ["项目", "满分", "实际扣分", "预测扣分", "实际得分", "预测得分", "分差"]

        self.tree = ttk.Treeview(root, columns=columns, show="headings", height=8)
        self.tree.pack(fill="both", expand=True, padx=20, pady=16)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=130)

        self.status_text = tk.Text(root, height=12, font=("Consolas", 10))
        self.status_text.pack(fill="x", padx=20, pady=8)

        self.log("请选择一个农户文件夹，例如 data/raw/97。")
        self.log("评分完成后，会生成 annotated 文字标注图和 gradcam 热力图。")

    def safe_call(self, func, *args, **kwargs):
        self.root.after(0, lambda: func(*args, **kwargs))

    def log(self, message):
        self.status_text.insert("end", str(message) + "\n")
        self.status_text.see("end")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="选择农户文件夹")

        if folder:
            self.selected_folder = Path(folder)
            self.folder_label.config(text=f"当前文件夹：{self.selected_folder}")
            self.log(f"已选择文件夹：{self.selected_folder}")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def run_score_thread(self):
        if self.selected_folder is None:
            messagebox.showwarning("提示", "请先选择农户文件夹")
            return

        thread = threading.Thread(target=self.run_score, daemon=True)
        thread.start()

    def format_value(self, value):
        if pd.isna(value):
            return "缺失"

        try:
            if float(value).is_integer():
                return str(int(value))
        except Exception:
            pass

        return str(value)

    def insert_results(self, result_df):
        self.clear_table()

        for _, row in result_df.iterrows():
            values = [
                row["项目"],
                self.format_value(row["满分"]),
                self.format_value(row["实际扣分"]),
                self.format_value(row["预测扣分"]),
                self.format_value(row["实际得分"]),
                self.format_value(row["预测得分"]),
                self.format_value(row["分差"]),
            ]

            self.tree.insert("", "end", values=values)

    def run_score(self):
        try:
            self.safe_call(self.clear_table)
            self.safe_call(self.log, "开始评分，请稍等...")

            result_df, label_csv, save_path, annotated_dir, gradcam_dir, annotated_paths, gradcam_paths, device = score_folder(
                self.selected_folder,
                log_fn=lambda msg: self.safe_call(self.log, msg)
            )

            self.annotated_dir = annotated_dir
            self.gradcam_dir = gradcam_dir

            total_row = result_df[result_df["项目"] == "总分"].iloc[0]

            self.safe_call(self.insert_results, result_df)
            self.safe_call(self.log, f"人工标签CSV：{label_csv}")
            self.safe_call(self.log, f"运行设备：{device}")
            self.safe_call(self.log, "评分完成。")
            self.safe_call(self.log, f"实际总分：{self.format_value(total_row['实际得分'])} / 60")
            self.safe_call(self.log, f"预测总分：{self.format_value(total_row['预测得分'])} / 60")
            self.safe_call(self.log, f"总分分差：{self.format_value(total_row['分差'])}")
            self.safe_call(self.log, f"结果已保存：{save_path}")
            self.safe_call(self.log, f"标注图文件夹：{annotated_dir}")
            self.safe_call(self.log, f"热力图文件夹：{gradcam_dir}")

        except Exception as e:
            self.safe_call(self.log, f"错误：{e}")
            self.safe_call(messagebox.showerror, "运行错误", str(e))

    def open_annotated_folder(self):
        if self.annotated_dir is None:
            messagebox.showwarning("提示", "还没有生成标注图，请先开始评分")
            return

        if not self.annotated_dir.exists():
            messagebox.showwarning("提示", f"标注图文件夹不存在：{self.annotated_dir}")
            return

        try:
            os.startfile(self.annotated_dir)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def open_gradcam_folder(self):
        if self.gradcam_dir is None:
            messagebox.showwarning("提示", "还没有生成热力图，请先开始评分")
            return

        if not self.gradcam_dir.exists():
            messagebox.showwarning("提示", f"热力图文件夹不存在：{self.gradcam_dir}")
            return

        try:
            os.startfile(self.gradcam_dir)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))


def main():
    root = tk.Tk()
    ScoreApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

