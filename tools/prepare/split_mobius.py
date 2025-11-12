#!/usr/bin/env python3
"""
Add train/val/test split column to a CSV using ID-level splits,
stratified within groups (e.g., by 'eye').

- Keeps all rows for a given `id` in the same split.
- Targets per-group ID ratios (defaults: train=0.80, val=0.10, test=0.10).
- Deterministic with --seed.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

SPLIT_TRAIN = "train"
SPLIT_VAL   = "val"
SPLIT_TEST  = "test"

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO   = 0.10
DEFAULT_TEST_RATIO  = 0.10

DEFAULT_STRATIFY_BY = "colour"

def _allocate_counts(n: int, ratios: list[float]) -> list[int]:
    """
    Allocate integer counts that sum to n based on ratios.
    Uses largest-remainder method for stable rounding.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("Sum of ratios must be > 0")
    shares = [r / total * n for r in ratios]
    floors = [int(np.floor(x)) for x in shares]
    leftover = n - sum(floors)
    # Distribute leftovers by largest fractional parts
    fracs = np.array([s - f for s, f in zip(shares, floors)])
    order = np.argsort(-fracs)  # descending
    for i in order[:leftover]:
        floors[i] += 1
    return floors

def add_split_column_by_id_stratified(
    csv_path: str,
    output_path: str | None = None,
    id_col: str = "id",
    stratify_by: list[str] = ["eye"],
    train_ratio: float = 0.80,
    val_ratio: float   = 0.10,
    test_ratio: float  = 0.10,
    seed: int = 42,
    min_test_per_group: int = 1, 
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    out_path = Path(output_path) if output_path else csv_path

    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise ValueError(f"Missing required id column: {id_col}")

    for col in stratify_by:
        if col not in df.columns:
            raise ValueError(f"Missing stratification column: {col}")

    rng = np.random.RandomState(seed)

    # Prepare split map for IDs
    id_to_split: dict[int, str] = {}

    # Work per group to target ratios within each grouping
    grouped = df.groupby(stratify_by, dropna=False, sort=False)

    # For nice reporting
    group_reports = []

    for group_key, gdf in grouped:
        # Unique IDs that appear in this group
        ids = pd.Index(gdf[id_col].unique())
        n_ids = len(ids)
        if n_ids == 0:
            continue

        # Shuffle deterministically
        ids = ids.to_numpy()
        rng.shuffle(ids)

        # Remove IDs already assigned (if an ID appears in multiple groups)
        unassigned_mask = np.array([i not in id_to_split for i in ids], dtype=bool)
        ids_unassigned = ids[unassigned_mask]
        n_unassigned = len(ids_unassigned)

        # Compute counts for this group based on unassigned IDs
        tr, vr, te = train_ratio, val_ratio, test_ratio
        counts = _allocate_counts(n_unassigned, [tr, vr, te])

        # --- NEW: enforce at least one test per group (when n_unassigned >= 1) ---
        counts = enforce_min_per_group(
            counts,
            n_total=n_unassigned,
            min_train=0,
            min_val=0,
            min_test=min_test_per_group,   # from CLI (default 1)
            reduction_priority=(0, 1, 2),  # prefer taking from train, then val, then test
        )

        n_train, n_val, n_test = counts

        # Assign
        start = 0
        for split_name, count in zip([SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL], [n_train, n_test, n_val]):
            for i in ids_unassigned[start:start+count]:
                id_to_split[int(i)] = split_name
            start += count

        # Reporting for this group
        group_reports.append({
            "group": group_key if isinstance(group_key, tuple) else (group_key,),
            "total_ids_in_group": n_ids,
            "unassigned_ids_in_group": n_unassigned,
            "assigned_train": n_train,
            "assigned_val": n_val,
            "assigned_test": n_test,
        })

    # Any remaining IDs that never showed up in grouped (shouldn't happen) -> put in train
    all_ids = set(df[id_col].unique().tolist())
    for i in all_ids:
        if i not in id_to_split:
            id_to_split[int(i)] = SPLIT_TRAIN

    # Map back to rows
    df["split"] = df[id_col].map(id_to_split)

    # ======= Reporting =======
    total_rows = len(df)
    total_ids = len(all_ids)
    split_row_ct = Counter(df["split"])
    split_id_ct = Counter([id_to_split[i] for i in all_ids])

    print("\n=== Overall (by ROWS) ===")
    for s in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        c = split_row_ct.get(s, 0)
        print(f"{s:>5}: {c:7d}  ({100.0*c/total_rows:5.1f}%)")

    print("\n=== Overall (by UNIQUE IDs) ===")
    for s in [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]:
        c = split_id_ct.get(s, 0)
        print(f"{s:>5}: {c:7d}  ({100.0*c/total_ids:5.1f}%)")

    # Per-group ID stats
    if group_reports:
        print("\n=== Per-Group ID Allocation (stratified by: " + ", ".join(stratify_by) + ") ===")
        for rep in group_reports:
            gkey = " | ".join(map(str, rep["group"]))
            print(
                f"[{gkey}] total_ids={rep['total_ids_in_group']:4d} "
                f"unassigned_seen_here={rep['unassigned_ids_in_group']:4d}  -> "
                f"train={rep['assigned_train']:4d} val={rep['assigned_val']:4d} test={rep['assigned_test']:4d}"
            )

    # Warn if IDs spanned multiple groups (we handled it by first-come assignment)
    # We detect multi-group IDs by counting distinct group keys per ID.
    id_group_counts = (
        df.groupby(id_col)[stratify_by].nunique() if len(stratify_by) == 1
        else df.groupby(id_col)[stratify_by].nunique().apply(tuple, axis=1)
    )
    # A simple heuristic: if any ID shows >1 distinct value in any strat column
    multi_group_ids = []
    if len(stratify_by) == 1:
        col = stratify_by[0]
        multi_group_ids = id_group_counts[id_group_counts > 1].index.tolist()
    else:
        # For multi-column strat, check if any column has >1 distinct value for an ID
        nunique_per_id = df.groupby(id_col)[stratify_by].nunique()
        multi_group_ids = nunique_per_id[(nunique_per_id > 1).any(axis=1)].index.tolist()

    if multi_group_ids:
        print(
            f"\n[warning] {len(multi_group_ids)} IDs appear in multiple stratification groups; "
            "they were assigned once (first group claimed them) to keep ID integrity."
        )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    return df


def enforce_min_per_group(
    counts: list[int],
    n_total: int,
    min_train: int = 0,
    min_val: int = 0,
    min_test: int = 1,
    reduction_priority: tuple[int, ...] = (0, 1, 2),  # 0=train,1=val,2=test
) -> list[int]:
    """
    Ensure per-group minimums while keeping sum(counts) == n_total.
    By default enforces at least one in TEST if n_total >= 1.

    Reduction priority is the order of buckets to subtract from
    when we must reduce due to raising some bucket to its minimum.
    """
    counts = list(map(int, counts))
    mins = [int(min_train), int(min_val), int(min_test)]

    # Edge case: no IDs to assign
    if n_total <= 0:
        return [0, 0, 0]

    # Raise buckets to their minimums
    for i in range(3):
        if counts[i] < mins[i]:
            counts[i] = mins[i]

    # If we exceeded n_total, reduce from other buckets by priority (without violating mins)
    excess = sum(counts) - n_total
    if excess > 0:
        for idx in reduction_priority:
            if excess == 0:
                break
            # how much can we take without going below its min?
            can_take = counts[idx] - mins[idx]
            if can_take <= 0:
                continue
            take = min(can_take, excess)
            counts[idx] -= take
            excess -= take

    # If we’re still in excess (mins sum > n_total), we must reduce even below mins;
    # try again by priority (this is the only way to resolve impossible demands).
    if excess > 0:
        for idx in reduction_priority:
            if excess == 0:
                break
            can_take = counts[idx]  # can go below min now
            if can_take <= 0:
                continue
            take = min(can_take, excess)
            counts[idx] -= take
            excess -= take

    # If we’re short (sum < n_total), add the remainder to TRAIN by default
    short = n_total - sum(counts)
    if short > 0:
        counts[0] += short

    # Final clamp to non-negative integers, exact sum
    counts = [max(0, int(x)) for x in counts]
    # (Optional assert) assert sum(counts) == n_total
    return counts

def main():
    p = argparse.ArgumentParser(description="Add ID-level stratified train/val/test split to CSV.")
    p.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    p.add_argument("--output", type=str, help="Path to output CSV (default: overwrite input)")
    p.add_argument("--id-col", type=str, default="ID", help="Column with identity ID")
    p.add_argument(
        "--stratify-by",
        type=str,
        default=DEFAULT_STRATIFY_BY,
        help="Comma-separated column(s) to stratify by (e.g., 'eye' or 'eye_code' or 'eye,light')",
    )
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train ratio (IDs)")
    p.add_argument("--val-ratio",   type=float, default=DEFAULT_VAL_RATIO, help="Val ratio (IDs)")
    p.add_argument("--test-ratio",  type=float, default=DEFAULT_TEST_RATIO, help="Test ratio (IDs)")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    p.add_argument("--min-test-per-group", type=int, default=1,
                help="Minimum number of IDs assigned to TEST in each strat group (default: 1).")
    args = p.parse_args()

    # Basic sanity on ratios
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        print(f"[note] Ratios sum to {ratios_sum:.6f}, will be normalized per group.")

    strat_cols = [s.strip() for s in args.stratify_by.split(",") if s.strip()]


    # pass it through:
    add_split_column_by_id_stratified(
        csv_path=args.csv,
        output_path=args.output,
        id_col=args.id_col,
        stratify_by=strat_cols,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_test_per_group=args.min_test_per_group,  # <---
    )


if __name__ == "__main__":
    main()
