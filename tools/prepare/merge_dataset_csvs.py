#!/usr/bin/env python3
"""
Merge multiple dataset CSVs into a single file suitable for IrisPupilNet training.

By default the script looks for the prepared datasets shipped in `dataset/`:
    - dataset/mobius_output/mobius_dataset_split.csv
    - dataset/tayed_output/tayed_dataset.csv
    - dataset/irispupileye_output/irispupileye_dataset.csv

Each source CSV must provide, at minimum, the columns required by the training
pipeline: `rel_image_path`, `rel_mask_path`, `split`. Missing metadata columns
(`id`, `dataset`, `dataset_format`) are filled automatically.

Usage:
    python tools/prepare/merge_dataset_csvs.py \
        --output_csv dataset/merged/combined_dataset.csv

    # Specify explicit list (overrides defaults)
    python tools/prepare/merge_dataset_csvs.py \
        --input_csv dataset/tayed_output/tayed_dataset.csv \
        --input_csv dataset/irispupileye_output/irispupileye_dataset.csv \
        --output_csv dataset/merged/tayed_irispupil.csv
"""

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

DEFAULT_INPUTS = [
    Path("dataset/mobius_output/mobius_dataset_split.csv"),
    Path("dataset/tayed_output/tayed_dataset.csv"),
    Path("dataset/irispupileye_output/irispupileye_dataset.csv"),
]

REQUIRED_COLUMNS = {"rel_image_path", "rel_mask_path", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge prepared dataset CSVs into a single file"
    )
    parser.add_argument(
        "--input_csv",
        action="append",
        dest="input_csvs",
        help="CSV to include (repeat flag for multiple). "
        "Defaults to the prepared CSVs under dataset/.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataset/merged/combined_dataset.csv",
        help="Path to the merged CSV to create",
    )
    parser.add_argument(
        "--drop_duplicates",
        action="store_true",
        help="Drop duplicate rows based on rel_image_path + rel_mask_path",
    )
    return parser.parse_args()


def infer_dataset_name(csv_path: Path) -> str:
    """
    Use the parent directory name as dataset identifier, stripping `_output`
    if present.
    """
    parent_name = csv_path.parent.name
    if parent_name.endswith("_output"):
        parent_name = parent_name[: -len("_output")]
    return parent_name


def ensure_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )


def standardize_dataframe(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    df = df.copy()
    ensure_required_columns(df, csv_path)

    dataset_name = infer_dataset_name(csv_path)

    if "dataset" not in df.columns:
        df["dataset"] = dataset_name

    if "dataset_format" not in df.columns:
        df["dataset_format"] = dataset_name

    if "id" not in df.columns:
        df["id"] = [
            f"{dataset_name}_{idx}" for idx in range(len(df))
        ]

    if "width" not in df.columns:
        df["width"] = pd.NA
    if "height" not in df.columns:
        df["height"] = pd.NA

    df["source_csv"] = str(csv_path)
    return df


def load_input_csvs(paths: List[Path]) -> Dict[Path, pd.DataFrame]:
    datasets: Dict[Path, pd.DataFrame] = {}
    for csv_path in paths:
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
        df = pd.read_csv(csv_path)
        df = standardize_dataframe(df, csv_path)
        datasets[csv_path] = df
        print(f"Loaded {len(df)} rows from {csv_path}")
    return datasets


def merge_dataframes(datasets: Dict[Path, pd.DataFrame], drop_duplicates: bool) -> pd.DataFrame:
    if not datasets:
        raise ValueError("No CSVs were loaded. Nothing to merge.")

    merged = pd.concat(datasets.values(), ignore_index=True, sort=False)

    if drop_duplicates:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["rel_image_path", "rel_mask_path"])
        after = len(merged)
        print(f"Dropped {before - after} duplicate rows.")

    return merged


def main():
    args = parse_args()
    input_csvs = (
        [Path(p) for p in args.input_csvs]
        if args.input_csvs
        else DEFAULT_INPUTS
    )

    datasets = load_input_csvs(input_csvs)
    merged_df = merge_dataframes(datasets, args.drop_duplicates)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved merged CSV to {output_path}")
    print(f"  Total rows: {len(merged_df)}")
    print("  Breakdown by dataset:")
    print(merged_df.groupby("dataset")["id"].count())
    print("\n  Breakdown by split:")
    print(merged_df.groupby("split")["id"].count())


if __name__ == "__main__":
    main()
