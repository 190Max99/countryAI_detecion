import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_safely(csv_path: Path):
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]

    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:
            pass

    raise RuntimeError(f"无法读取 CSV 文件: {csv_path}")


def collect_score_results(root_dir: Path):
    """
    扫描 data/raw 下每个农户文件夹中的 ui_score_result_xxx.csv
    汇总为一个总表。
    """

    all_rows = []

    for folder in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not folder.is_dir():
            continue

        result_files = sorted(folder.glob("ui_score_result_*.csv"))

        if len(result_files) == 0:
            continue

        result_file = result_files[0]

        try:
            df = read_csv_safely(result_file)
        except Exception as e:
            print(f"跳过 {result_file}: {e}")
            continue

        house_id = folder.name

        for _, row in df.iterrows():
            item = str(row.get("项目", ""))

            true_score = row.get("实际得分", np.nan)
            pred_score = row.get("预测得分", np.nan)
            error = row.get("分差", np.nan)

            try:
                true_score = float(true_score)
            except Exception:
                true_score = np.nan

            try:
                pred_score = float(pred_score)
            except Exception:
                pred_score = np.nan

            try:
                error = float(error)
            except Exception:
                error = np.nan

            all_rows.append({
                "house_id": house_id,
                "项目": item,
                "实际得分": true_score,
                "预测得分": pred_score,
                "分差": error,
                "绝对误差": abs(error) if not pd.isna(error) else np.nan,
                "result_file": str(result_file),
            })

    result_df = pd.DataFrame(all_rows)

    return result_df


def plot_total_score_curve(df: pd.DataFrame, output_dir: Path):
    """
    绘制每户实际总分 vs 预测总分曲线。
    """

    total_df = df[df["项目"] == "总分"].copy()

    if len(total_df) == 0:
        print("没有找到总分数据，无法绘制总分曲线")
        return

    total_df = total_df.sort_values("house_id")

    x = np.arange(len(total_df))

    plt.figure(figsize=(12, 6))
    plt.plot(x, total_df["实际得分"], marker="o", label="实际总分")
    plt.plot(x, total_df["预测得分"], marker="s", label="预测总分")

    plt.xticks(x, total_df["house_id"], rotation=45)
    plt.xlabel("农户编号")
    plt.ylabel("总分")
    plt.title("每户实际总分与 AI 预测总分对比")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / "total_score_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已保存：{save_path}")


def plot_total_error_curve(df: pd.DataFrame, output_dir: Path):
    """
    绘制每户总分分差曲线。
    分差 = 预测得分 - 实际得分
    """

    total_df = df[df["项目"] == "总分"].copy()

    if len(total_df) == 0:
        print("没有找到总分数据，无法绘制误差曲线")
        return

    total_df = total_df.sort_values("house_id")

    x = np.arange(len(total_df))

    plt.figure(figsize=(12, 6))
    plt.plot(x, total_df["分差"], marker="o", label="总分分差")
    plt.axhline(y=0, linestyle="--", label="零误差线")

    plt.xticks(x, total_df["house_id"], rotation=45)
    plt.xlabel("农户编号")
    plt.ylabel("分差：预测得分 - 实际得分")
    plt.title("每户总分误差曲线")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / "total_error_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已保存：{save_path}")


def plot_scene_mae_bar(df: pd.DataFrame, output_dir: Path):
    """
    绘制各场景平均绝对误差 MAE。
    """

    scene_df = df[df["项目"] != "总分"].copy()

    if len(scene_df) == 0:
        print("没有找到场景数据，无法绘制 MAE 图")
        return

    mae_df = (
        scene_df
        .groupby("项目", as_index=False)["绝对误差"]
        .mean()
        .rename(columns={"绝对误差": "MAE"})
    )

    plt.figure(figsize=(10, 6))
    plt.bar(mae_df["项目"], mae_df["MAE"])

    plt.xlabel("场景")
    plt.ylabel("平均绝对误差 MAE")
    plt.title("各场景评分误差对比")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / "scene_mae_bar.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已保存：{save_path}")


def plot_scene_error_curve(df: pd.DataFrame, output_dir: Path):
    """
    绘制各场景分差曲线。
    """

    scene_names = ["室内", "庭院", "厕所及化粪池", "房前屋后"]

    plt.figure(figsize=(12, 6))

    for scene in scene_names:
        scene_df = df[df["项目"] == scene].copy()

        if len(scene_df) == 0:
            continue

        scene_df = scene_df.sort_values("house_id")
        x = np.arange(len(scene_df))

        plt.plot(x, scene_df["分差"], marker="o", label=scene)

    house_ids = sorted(df[df["项目"] == "总分"]["house_id"].unique())

    if len(house_ids) > 0:
        plt.xticks(np.arange(len(house_ids)), house_ids, rotation=45)

    plt.axhline(y=0, linestyle="--", label="零误差线")
    plt.xlabel("农户编号")
    plt.ylabel("分差：预测得分 - 实际得分")
    plt.title("各场景评分误差曲线")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / "scene_error_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已保存：{save_path}")


def plot_accuracy_by_tolerance(df: pd.DataFrame, output_dir: Path):
    """
    绘制总分容差准确率。

    例如：
    分差绝对值 <= 0：完全一致
    分差绝对值 <= 2：误差在 2 分以内
    分差绝对值 <= 5：误差在 5 分以内
    """

    total_df = df[df["项目"] == "总分"].copy()

    if len(total_df) == 0:
        print("没有找到总分数据，无法绘制容差准确率")
        return

    tolerances = [0, 1, 2, 3, 5, 10]
    accuracies = []

    for tol in tolerances:
        acc = (total_df["绝对误差"] <= tol).mean()
        accuracies.append(acc)

    plt.figure(figsize=(9, 6))
    plt.plot(tolerances, accuracies, marker="o")

    plt.xlabel("允许误差范围")
    plt.ylabel("准确率")
    plt.title("总分容差准确率曲线")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = output_dir / "score_accuracy_tolerance_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"已保存：{save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="data/raw",
        help="农户文件夹根目录，例如 data/raw"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/curves",
        help="曲线图输出目录"
    )

    args = parser.parse_args()

    root_dir = Path(args.root)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_score_results(root_dir)

    if len(df) == 0:
        print("没有找到任何 ui_score_result_*.csv，请先运行 UI 对农户文件夹评分。")
        return

    summary_path = output_dir / "score_error_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"已保存汇总表：{summary_path}")

    plot_total_score_curve(df, output_dir)
    plot_total_error_curve(df, output_dir)
    plot_scene_mae_bar(df, output_dir)
    plot_scene_error_curve(df, output_dir)
    plot_accuracy_by_tolerance(df, output_dir)

    print("全部曲线绘制完成。")


if __name__ == "__main__":
    main()