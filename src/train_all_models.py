"""Train all scene models after a scene-type precheck."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from src.scene_type_precheck import (
    DEFAULT_CLASSIFIER,
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_REPORT,
    check_scene_types,
    read_csv_safely,
)


TRAIN_JOBS = [
    {
        "name": "indoor",
        "title": "indoor",
        "module": "src.train_test_indoor",
        "supports_train_num": False,
        "extra_args": [],
    },
    {
        "name": "outside",
        "title": "outside with curves",
        "module": "src.train_outside_70_with_curves",
        "supports_train_num": True,
        "extra_args": [],
    },
    {
        "name": "courtyard",
        "title": "courtyard",
        "module": "src.train_courtyard_70",
        "supports_train_num": True,
        "extra_args": [],
    },
    {
        "name": "courtyard_cbam",
        "title": "courtyard CBAM",
        "module": "src.train_courtyard_cbam_70",
        "supports_train_num": True,
        "extra_args": [],
    },
    {
        "name": "toilet",
        "title": "toilet",
        "module": "src.train_toilet_70",
        "supports_train_num": True,
        "extra_args": [],
    },
    {
        "name": "septic",
        "title": "septic",
        "module": "src.train_septic_70",
        "supports_train_num": True,
        "extra_args": [],
    },
]


def parse_name_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_command(job: dict, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        job["module"],
        "--csv",
        args.csv,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]

    if job["supports_train_num"]:
        command.extend(["--train_num", str(args.train_num)])

    command.extend(job["extra_args"])
    return command


def select_jobs(args: argparse.Namespace) -> list[dict]:
    all_names = {job["name"] for job in TRAIN_JOBS}
    only_names = set(parse_name_list(args.only))
    skip_names = set(parse_name_list(args.skip))

    unknown_names = (only_names | skip_names) - all_names
    if unknown_names:
        raise ValueError(
            "Unknown model name: "
            + ", ".join(sorted(unknown_names))
            + "\nAvailable names: "
            + ", ".join(job["name"] for job in TRAIN_JOBS)
        )

    selected = []
    for job in TRAIN_JOBS:
        if only_names and job["name"] not in only_names:
            continue
        if job["name"] in skip_names:
            continue
        selected.append(job)

    if not selected:
        raise ValueError("No model selected. Check --only or --skip.")

    return selected


def write_filtered_training_csv(args: argparse.Namespace, summary: dict) -> str:
    source_csv = Path(args.csv)
    report_csv = Path(summary["report_path"])
    filtered_csv = Path(args.filtered_csv)

    source_df = read_csv_safely(source_csv)
    report_df = read_csv_safely(report_csv)

    bad_rows = report_df[report_df["high_confidence_mismatch"].astype(int) == 1]
    bad_indices = set(bad_rows["row_index"].astype(int).tolist())

    filtered_df = source_df.drop(index=list(bad_indices), errors="ignore")
    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(filtered_csv, index=False, encoding="utf-8-sig")

    print("Removed high-confidence wrong-scene rows:", len(bad_indices))
    if bad_indices:
        print("Filtered training CSV:", filtered_csv)
        print("Removed images:")
        for _, row in bad_rows.iterrows():
            print(
                f"- row {row['row_index']}: {row.get('expected_scene', '')} -> "
                f"{row.get('pred_scene', '')}, confidence={float(row.get('confidence', 0)):.3f}, "
                f"image={row.get('image_path', '')}"
            )

    return str(filtered_csv) if bad_indices else str(source_csv)


def run_scene_precheck(args: argparse.Namespace) -> None:
    if args.skip_scene_check:
        print("\nScene precheck skipped.")
        return

    print("\n========== Scene type precheck ==========")
    summary = check_scene_types(
        csv_path=args.csv,
        classifier_path=args.scene_classifier,
        feature_weights_path=args.scene_feature_weights,
        report_path=args.scene_check_report,
        confidence_threshold=args.scene_confidence_threshold,
    )

    print("Total rows:", summary["total"])
    print("Image errors:", summary["error_count"])
    print("Mismatches:", summary["mismatch_count"])
    print("High-confidence mismatches:", summary["high_confidence_mismatch_count"])
    print("Report:", summary["report_path"])

    if summary["error_count"] > 0 and not args.allow_scene_mismatch:
        print("\nScene precheck found image errors. Please review the report before training.")
        raise SystemExit(1)

    if summary["high_confidence_mismatch_count"] > 0:
        if args.allow_scene_mismatch:
            print("\nHigh-confidence mismatches found, but training will continue with the original CSV.")
            return
        args.csv = write_filtered_training_csv(args, summary)
        sys.stdout.flush()
        return

    print("Scene precheck passed. No high-confidence wrong-scene images found.")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all scene models in sequence.")
    parser.add_argument("--csv", default="data/all_labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_num", type=int, default=70)
    parser.add_argument(
        "--only",
        default="",
        help="Only train selected models, comma-separated. Options: indoor,outside,courtyard,courtyard_cbam,toilet,septic",
    )
    parser.add_argument("--skip", default="", help="Skip selected models, comma-separated.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue after one training job fails.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--skip_scene_check",
        action="store_true",
        help="Skip scene classifier precheck before training.",
    )
    parser.add_argument(
        "--allow_scene_mismatch",
        action="store_true",
        help="Train with the original CSV even when high-confidence scene mismatches are found.",
    )
    parser.add_argument("--scene_classifier", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--scene_feature_weights", default=str(DEFAULT_FEATURE_WEIGHTS))
    parser.add_argument("--scene_check_report", default=str(DEFAULT_REPORT))
    parser.add_argument("--scene_confidence_threshold", type=float, default=0.70)
    parser.add_argument("--filtered_csv", default="outputs/all_labels_scene_checked.csv")

    args = parser.parse_args()
    jobs = select_jobs(args)

    if not args.dry_run:
        run_scene_precheck(args)

    print("\n========== Train all models ==========")
    print("CSV:", args.csv)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("lr:", args.lr)
    print("train_num:", args.train_num)
    print("Jobs:", " -> ".join(job["title"] for job in jobs))
    sys.stdout.flush()

    failures = []
    start_time = time.time()

    for index, job in enumerate(jobs, start=1):
        command = build_command(job, args)

        print("\n" + "=" * 60)
        print(f"[{index}/{len(jobs)}] Start: {job['title']}")
        print("Command:", " ".join(command))
        print("=" * 60)
        sys.stdout.flush()

        if args.dry_run:
            continue

        job_start = time.time()
        result = subprocess.run(command)
        elapsed = time.time() - job_start

        if result.returncode == 0:
            print(f"\nDone: {job['title']}, elapsed {elapsed / 60:.1f} minutes")
            continue

        failures.append((job, result.returncode))
        print(f"\nFailed: {job['title']}, exit code {result.returncode}, elapsed {elapsed / 60:.1f} minutes")

        if not args.continue_on_error:
            break

    total_elapsed = time.time() - start_time

    print("\n========== Training finished ==========")
    print(f"Total elapsed: {total_elapsed / 60:.1f} minutes")

    if failures:
        print("Failed jobs:")
        for job, returncode in failures:
            print(f"- {job['title']} ({job['name']}), exit code {returncode}")
        raise SystemExit(1)

    print("All selected models finished.")


if __name__ == "__main__":
    main()
