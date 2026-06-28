# build_dataset.py

import argparse
from pathlib import Path

import pandas as pd

from config import SCENE_CONFIG


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_csv_file(folder: Path):
    """
    在每个农户文件夹中寻找 CSV 标注文件。
    例如：
    data/raw/6/6.csv
    """
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        return None

    return csv_files[0]


def read_csv_safely(csv_path: Path):
    """
    自动尝试多种编码读取 CSV，避免中文乱码或读取失败。
    """
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]

    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return df
        except Exception:
            continue

    raise ValueError(f"CSV 文件读取失败，请检查编码或文件格式: {csv_path}")


def find_image_file(folder: Path, scene: str, house_id: str):
    """
    根据场景和农户编号寻找对应图片。
    支持：
    室内_6.jpg
    6_室内.jpg
    或文件名里包含“室内”的图片
    """
    candidates = []

    for ext in IMAGE_EXTS:
        candidates.append(folder / f"{scene}_{house_id}{ext}")
        candidates.append(folder / f"{house_id}_{scene}{ext}")

    for p in candidates:
        if p.exists():
            return p

    for ext in IMAGE_EXTS:
        matches = list(folder.glob(f"*{scene}*{ext}"))
        if matches:
            return matches[0]

    return None


def clean_label_value(value):
    """
    把 CSV 中的标签值转成 0 或 1。
    空白、NaN 都按 0 处理。
    """
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    try:
        return int(float(value))
    except Exception:
        return 0


def read_one_house(folder: Path):
    house_id = folder.name
    csv_path = find_csv_file(folder)

    if csv_path is None:
        print(f"[跳过] {folder} 没有找到 CSV 标注文件")
        return []

    try:
        df = read_csv_safely(csv_path)
    except Exception as e:
        print(f"[跳过] 读取失败: {csv_path}, 原因: {e}")
        return []

    if df.empty:
        print(f"[跳过] 空表: {csv_path}")
        return []

    # 第一列一般是“序号”，里面写着 厕所、房前屋后、室内、庭院、化粪池
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "scene"})

    rows = []

    for _, row in df.iterrows():
        scene = str(row.get("scene", "")).strip()

        if scene not in SCENE_CONFIG:
            continue

        image_path = find_image_file(folder, scene, house_id)

        if image_path is None:
            print(f"[警告] {folder} 未找到场景图片: {scene}")
            continue

        num_labels = SCENE_CONFIG[scene]["num_labels"]

        item = {
            "house_id": house_id,
            "scene": scene,
            "image_path": str(image_path).replace("\\", "/")
        }

        for i in range(12):
            col = f"label_{i}"

            if col in df.columns:
                value = clean_label_value(row[col])
            else:
                value = 0

            if i < num_labels:
                item[col] = value
            else:
                item[col] = ""

        rows.append(item)

    return rows


def build_dataset(raw_dir: str, out_csv: str):
    raw_path = Path(raw_dir)
    all_rows = []

    for folder in sorted(raw_path.iterdir()):
        if folder.is_dir():
            rows = read_one_house(folder)
            all_rows.extend(rows)

    if not all_rows:
        print("没有整理出任何数据，请检查 data/raw 目录结构")
        return

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(all_rows)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n整理完成: {out_path}")
    print(f"总图片数量: {len(result)}")

    print("\n各场景数量:")
    print(result["scene"].value_counts())

    print("\n前几行数据预览:")
    print(result.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/all_labels.csv")
    args = parser.parse_args()

    build_dataset(args.raw_dir, args.out)