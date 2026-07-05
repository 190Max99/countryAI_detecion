"""
Pre-check image scene types before model training.

The checker reads data/all_labels.csv, predicts each image's scene with the
external scene classifier, writes a report, and returns mismatches.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms


DEFAULT_CLASSIFIER = Path("models/scene_classifier_resnet18.pkl")
DEFAULT_FEATURE_WEIGHTS = Path("models/scene_resnet18_feature_extractor.pth")
DEFAULT_REPORT = Path("outputs/scene_type_precheck.csv")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCENE_ALIASES = {
    "室内": "室内",
    "屋内": "室内",
    "庭院": "庭院",
    "院内": "庭院",
    "厕所": "厕所",
    "厕屋": "厕所",
    "化粪池": "化粪池",
    "房前屋后": "房前屋后",
    "房前屋後": "房前屋后",
    "屋后": "房前屋后",
    "屋後": "房前屋后",
}


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, sep=None, engine="python")
        except Exception as exc:
            last_error = exc

    raise last_error


def normalize_scene(value) -> str:
    text = str(value).strip()
    if text in SCENE_ALIASES:
        return SCENE_ALIASES[text]

    for key, scene in SCENE_ALIASES.items():
        if key in text:
            return scene

    return text


def resolve_image_path(path_value, csv_path: Path) -> Path:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path

    project_path = Path.cwd() / path
    if project_path.exists():
        return project_path

    return csv_path.parent / path


def load_feature_extractor(feature_weights: Path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Identity()

    if feature_weights.exists():
        state = torch.load(feature_weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model, transform, device


def extract_feature(image_path: Path, model, transform, device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(x).detach().cpu().numpy()[0]
    return feature.astype(np.float32)


def check_scene_types(
    csv_path: str | Path = "data/all_labels.csv",
    classifier_path: str | Path = DEFAULT_CLASSIFIER,
    feature_weights_path: str | Path = DEFAULT_FEATURE_WEIGHTS,
    report_path: str | Path = DEFAULT_REPORT,
    confidence_threshold: float = 0.70,
) -> dict:
    csv_path = Path(csv_path)
    classifier_path = Path(classifier_path)
    feature_weights_path = Path(feature_weights_path)
    report_path = Path(report_path)

    if not classifier_path.exists():
        raise FileNotFoundError(f"Scene classifier not found: {classifier_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = read_csv_safely(csv_path)
    required_cols = {"scene", "image_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing_cols))}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(classifier_path)
    classifier = bundle["classifier"]
    classes = list(classifier.classes_)
    model, transform, device = load_feature_extractor(feature_weights_path)

    rows = []
    for index, row in df.iterrows():
        expected_scene = normalize_scene(row["scene"])
        image_path = resolve_image_path(row["image_path"], csv_path)

        out = {
            "row_index": index,
            "house_id": row.get("house_id", ""),
            "image_path": str(image_path),
            "expected_scene": expected_scene,
            "pred_scene": "",
            "confidence": 0.0,
            "is_match": 0,
            "high_confidence_mismatch": 0,
            "error": "",
        }

        try:
            if image_path.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"Unsupported image suffix: {image_path.suffix}")
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))

            feature = extract_feature(image_path, model, transform, device)
            pred = classifier.predict(feature.reshape(1, -1))[0]
            prob = classifier.predict_proba(feature.reshape(1, -1))[0]
            pred_scene = normalize_scene(pred)
            confidence = float(prob.max())
            is_match = int(expected_scene == pred_scene)

            out.update(
                {
                    "pred_scene": pred_scene,
                    "confidence": confidence,
                    "is_match": is_match,
                    "high_confidence_mismatch": int(
                        not is_match and confidence >= confidence_threshold
                    ),
                }
            )

            for scene in classes:
                out[f"prob_{normalize_scene(scene)}"] = float(prob[classes.index(scene)])
        except Exception as exc:
            out["error"] = str(exc)

        rows.append(out)

    report = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    error_count = int((report["error"].astype(str) != "").sum())
    mismatch_count = int((report["is_match"] == 0).sum())
    high_confidence_mismatch_count = int((report["high_confidence_mismatch"] == 1).sum())

    return {
        "total": len(report),
        "error_count": error_count,
        "mismatch_count": mismatch_count,
        "high_confidence_mismatch_count": high_confidence_mismatch_count,
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check scene type before training.")
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--feature_weights", default=str(DEFAULT_FEATURE_WEIGHTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--confidence_threshold", type=float, default=0.70)
    parser.add_argument(
        "--fail_on_mismatch",
        action="store_true",
        help="Exit with code 1 when high-confidence mismatches or image errors exist.",
    )
    args = parser.parse_args()

    summary = check_scene_types(
        csv_path=args.csv,
        classifier_path=args.classifier,
        feature_weights_path=args.feature_weights,
        report_path=args.report,
        confidence_threshold=args.confidence_threshold,
    )

    print("Scene precheck finished.")
    print("Total rows:", summary["total"])
    print("Image errors:", summary["error_count"])
    print("Mismatches:", summary["mismatch_count"])
    print("High-confidence mismatches:", summary["high_confidence_mismatch_count"])
    print("Report:", summary["report_path"])

    if args.fail_on_mismatch and (
        summary["error_count"] > 0 or summary["high_confidence_mismatch_count"] > 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
