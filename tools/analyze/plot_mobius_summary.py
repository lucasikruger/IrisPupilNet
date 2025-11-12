#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera tres summaries:
  1) summary_general.png  — barras simples (todo azul), % sobre total, conteo dentro; Images/ID con línea de media
  2) summary_groups.png   — barras agrupadas por sub-grupo (G1/G2/G3), % relativo a cada grupo, conteo dentro
  3) summary_splits.png   — NUEVO: barras agrupadas por split (train/val/test), % relativo a cada split, conteo dentro

Grupos por cantidad de imágenes por ID:
  - G1: ≤ 50
  - G2: 51-99
  - G3: ≥ 100

Usa SOLO columnas de nombres (no códigos): gender, colour, glasses/lenses, smoker, phone_model, light, eye, gaze.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG / STYLE
# ============================
DEFAULT_INPUT_CSV = "/home/agot-lkruger/tesis/IrisPupilNet/dataset/mobius_output/mobius_dataset_split.csv"
DEFAULT_OUTPUT_DIR = "/home/agot-lkruger/tesis/IrisPupilNet/dataset/mobius_output/plots"

# Azules para grupos por tamaño
BLUE    = "#1f77b4"  # general
BLUE_G1 = "#74a9cf"  # ≤50
BLUE_G2 = "#2b8cbe"  # 51-99
BLUE_G3 = "#1f77b4"  # ≥100

# Colores para splits
SPLIT_TRAIN_COLOR = "#2ca02c"  # green
SPLIT_VAL_COLOR   = "#ff7f0e"  # orange
SPLIT_TEST_COLOR  = "#9467bd"  # purple

LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 12

# Columnas categóricas (NOMBRES, no códigos)
CATEGORICAL_NAME_COLS = [
    "gender",
    "colour",
    "glasses/lenses",
    "smoker",
    "phone_model",
    "light",
    "eye",
    "gaze",
]

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_TO_COLOR = {
    "train": SPLIT_TRAIN_COLOR,
    "val": SPLIT_VAL_COLOR,
    "test": SPLIT_TEST_COLOR,
}

# ============================
# HELPERS
# ============================
def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )

def add_count_and_pct(ax, bars, values, pct_base, color_inside="white"):
    """Conteo dentro + % arriba. pct_base = denominador para el porcentaje."""
    total = float(pct_base) if pct_base else 1.0
    for rect, v in zip(bars, values):
        # Conteo dentro (centro)
        ax.text(
            rect.get_x() + rect.get_width()/2.0,
            rect.get_height()*0.5,
            f"{int(v)}",
            ha="center", va="center",
            color=color_inside, fontsize=LABEL_FONTSIZE, fontweight="bold"
        )
        # % arriba
        pct = 100.0 * (v / total) if total > 0 else 0.0
        ax.text(
            rect.get_x() + rect.get_width()/2.0,
            rect.get_height(),
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=LABEL_FONTSIZE
        )

def attach_group_by_id_count(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Agrega 'group' por ID según cantidad de imágenes: G1 ≤50, G2 51-99, G3 ≥100."""
    id_counts = df["ID"].value_counts()
    def tag(id_):
        n = int(id_counts.get(id_, 0))
        if n <= 50: return "G1 ≤50"
        if n <= 99: return "G2 51-99"
        return "G3 ≥100"
    out = df.copy()
    out["group"] = out["ID"].map(tag)
    return out, id_counts.sort_index()

def plot_images_per_id(ax, id_counts: pd.Series, color=BLUE, show_mean=True):
    ids = id_counts.index.to_list()
    vals = id_counts.values.astype(int)
    total = int(vals.sum())
    bars = ax.bar(ids, vals, color=color, edgecolor="#333", linewidth=0.5)
    add_count_and_pct(ax, bars, vals, pct_base=total, color_inside="white")
    if show_mean and len(vals):
        mean_val = float(np.mean(vals))
        ax.axhline(mean_val, color="#004c97", linestyle="--", linewidth=1.5)
        ax.text(0.995, 0.98, f"Mean: {mean_val:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.95, edgecolor="#888"))
    ax.set_title(f"Images per ID — Total: {total}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Images with mask")
    ax.set_xticks(ids)
    if len(vals): ax.set_ylim(0, max(vals) * 1.22)

def plot_hist(ax, values, color=BLUE):
    ax.hist(values, bins=10, color=color, edgecolor="#333")
    ax.set_title("Counts per Subject (Histogram)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Images per subject")
    ax.set_ylabel("Number of subjects")

def plot_box(ax, values):
    ax.boxplot(values, vert=True, tick_labels=["counts"])
    ax.set_title("Counts per Subject (Boxplot)", fontsize=TITLE_FONTSIZE)

def plot_cat_simple(ax, counts: pd.Series, label: str, color=BLUE, rotate_deg=None):
    labels = counts.index.astype(str).tolist()
    vals = counts.values.astype(int)
    total = int(sum(vals))
    bars = ax.bar(labels, vals, color=color, edgecolor="#333", linewidth=0.5)
    add_count_and_pct(ax, bars, vals, pct_base=total, color_inside="white")
    ax.set_title(f"{label} — Total: {total}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_ylim(0, (max(vals) * 1.22) if len(vals) else 1)

    rotation = rotate_deg if rotate_deg is not None else (15 if len(labels) > 5 else 0)
    ax.tick_params(axis="x", labelrotation=rotation)
    for t in ax.get_xticklabels():
        t.set_ha("right")

def plot_groups_summary(ax, id_counts: pd.Series):
    v = id_counts
    g1_total = int(v[v <= 50].sum())
    g2_total = int(v[(v >= 51) & (v <= 99)].sum())
    g3_total = int(v[v >= 100].sum())
    labels = ["≤50", "51-99", "≥100"]
    values = [g1_total, g2_total, g3_total]
    total_all = int(sum(values))
    bars = ax.bar(labels, values, color=[BLUE_G1, BLUE_G2, BLUE_G3], edgecolor="#333", linewidth=0.7)
    add_count_and_pct(ax, bars, values, pct_base=total_all, color_inside="white")
    ax.set_title(f"Images per ID — Group Summary — Total: {total_all}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Group")
    ax.set_ylabel("Images (sum)")
    if values: ax.set_ylim(0, max(values) * 1.22)

def plot_cat_grouped(ax, df_with_group: pd.DataFrame, column: str):
    """Barras agrupadas por categoría: 3 barras (G1,G2,G3) por cada categoría."""
    if column not in df_with_group.columns:
        ax.axis("off"); ax.set_title(f"{column} (not found)", fontsize=TITLE_FONTSIZE); return
    g1 = df_with_group[df_with_group["group"] == "G1 ≤50"]
    g2 = df_with_group[df_with_group["group"] == "G2 51-99"]
    g3 = df_with_group[df_with_group["group"] == "G3 ≥100"]
    N1, N2, N3 = int(len(g1)), int(len(g2)), int(len(g3))

    c1 = g1[column].dropna().value_counts()
    c2 = g2[column].dropna().value_counts()
    c3 = g3[column].dropna().value_counts()
    cats = sorted(set(c1.index.tolist()) | set(c2.index.tolist()) | set(c3.index.tolist()))
    if not cats:
        ax.axis("off"); ax.set_title(f"{column} (empty)", fontsize=TITLE_FONTSIZE); return

    v1 = [int(c1.get(cat, 0)) for cat in cats]
    v2 = [int(c2.get(cat, 0)) for cat in cats]
    v3 = [int(c3.get(cat, 0)) for cat in cats]

    width = 0.25
    x = np.arange(len(cats))
    b1 = ax.bar(x - width, v1, width, color=BLUE_G1, edgecolor="#333", linewidth=0.5, label=f"G1 ≤50 (N={N1})")
    b2 = ax.bar(x,         v2, width, color=BLUE_G2, edgecolor="#333", linewidth=0.5, label=f"G2 51-99 (N={N2})")
    b3 = ax.bar(x + width, v3, width, color=BLUE_G3, edgecolor="#333", linewidth=0.5, label=f"G3 ≥100 (N={N3})")

    add_count_and_pct(ax, b1, v1, pct_base=max(N1,1), color_inside="white")
    add_count_and_pct(ax, b2, v2, pct_base=max(N2,1), color_inside="white")
    add_count_and_pct(ax, b3, v3, pct_base=max(N3,1), color_inside="white")

    ax.set_title(f"{column} — Grouped (G1/G2/G3)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticks(x, cats, rotation=15 if len(cats) > 5 else 0)
    max_bar = max(v1 + v2 + v3) if (v1 or v2 or v3) else 1
    ax.set_ylim(0, max_bar * 1.55)
    ax.legend(loc="upper right", frameon=True)

# --------- NEW: SPLIT-BASED PLOTS ---------
def plot_splits_summary(ax, df: pd.DataFrame):
    """Resumen simple por split (train/val/test) con % sobre el total de filas."""
    counts = df["split"].value_counts()
    labels = [s for s in SPLIT_ORDER if s in counts.index]
    values = [int(counts.get(s, 0)) for s in labels]
    colors = [SPLIT_TO_COLOR[s] for s in labels]
    total = int(sum(values))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333", linewidth=0.7)
    add_count_and_pct(ax, bars, values, pct_base=total, color_inside="white")
    ax.set_title(f"Split Summary — Total rows: {total}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Split")
    ax.set_ylabel("Rows")
    if values: ax.set_ylim(0, max(values) * 1.22)

def plot_cat_grouped_by(ax, df: pd.DataFrame, column: str, group_col: str = "split"):
    """
    Barras agrupadas por `group_col` (p.ej. split).
    3 barras (train/val/test) por cada categoría de `column`.
    % relativo al tamaño del grupo (N de filas en ese split).
    """
    if column not in df.columns:
        ax.axis("off"); ax.set_title(f"{column} (not found)", fontsize=TITLE_FONTSIZE); return
    if group_col not in df.columns:
        ax.axis("off"); ax.set_title(f"{group_col} (not found)", fontsize=TITLE_FONTSIZE); return

    # Filas por split (denominadores)
    split_sizes = df[group_col].value_counts().to_dict()
    splits = [s for s in SPLIT_ORDER if s in split_sizes]  # keep order

    # Conteos por categoría dentro de cada split
    cats = sorted(set(df[column].dropna().unique().tolist()))
    if not cats:
        ax.axis("off"); ax.set_title(f"{column} (empty)", fontsize=TITLE_FONTSIZE); return

    width = 0.22 if len(splits) == 3 else (0.28 if len(splits) == 2 else 0.35)
    x = np.arange(len(cats))

    # Para calcular altura máx del eje Y
    all_vals = []

    for idx, split_name in enumerate(splits):
        sub = df[df[group_col] == split_name]
        c = sub[column].dropna().value_counts()
        vals = [int(c.get(cat, 0)) for cat in cats]
        offset = (-width if len(splits) == 3 else -width/2.0) + idx * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=SPLIT_TO_COLOR.get(split_name, "#888"),
            edgecolor="#333",
            linewidth=0.5,
            label=f"{split_name} (N={int(split_sizes.get(split_name, 0))})",
        )
        add_count_and_pct(ax, bars, vals, pct_base=max(int(split_sizes.get(split_name, 0)), 1), color_inside="white")
        all_vals.extend(vals)

    ax.set_title(f"{column} — Grouped by split", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticks(x, cats, rotation=15 if len(cats) > 5 else 0)
    max_bar = max(all_vals) if all_vals else 1
    ax.set_ylim(0, max_bar * 1.55)
    ax.legend(loc="upper right", frameon=True)
# --- NEW: grouped-by with dual bars (rows + unique IDs)
def plot_cat_grouped_by_dual(ax, df: pd.DataFrame, column: str, group_col: str = "split", id_col: str = "ID"):
    """
    For each category in `column`, draw two bars per split:
      - solid = row counts
      - hatched = unique ID counts
    """
    import matplotlib.patches as mpatches

    if column not in df.columns:
        ax.axis("off"); ax.set_title(f"{column} (not found)", fontsize=TITLE_FONTSIZE); return
    if group_col not in df.columns:
        ax.axis("off"); ax.set_title(f"{group_col} (not found)", fontsize=TITLE_FONTSIZE); return
    if id_col not in df.columns:
        ax.axis("off"); ax.set_title(f"{id_col} (not found)", fontsize=TITLE_FONTSIZE); return

    # Order splits and collect categories
    split_sizes = df[group_col].value_counts().to_dict()
    splits = [s for s in SPLIT_ORDER if s in split_sizes]
    cats = sorted(set(df[column].dropna().unique().tolist()))
    if not cats or not splits:
        ax.axis("off"); ax.set_title(f"{column} (empty)", fontsize=TITLE_FONTSIZE); return

    # Geometry: each category has len(splits) groups; each group has 2 bars (rows, ids)
    group_w = 0.22 if len(splits) == 3 else (0.28 if len(splits) == 2 else 0.35)
    bar_w   = group_w * 0.42  # each of the pair a bit narrower
    x = np.arange(len(cats))

    max_y = 1
    legend_handles = []
    first_rows_handle = None
    first_ids_handle  = None

    for idx, split_name in enumerate(splits):
        sub = df[df[group_col] == split_name]
        # row counts per category
        c_rows = sub[column].dropna().value_counts()
        v_rows = [int(c_rows.get(cat, 0)) for cat in cats]
        # unique ID counts per category
        v_ids = []
        for cat in cats:
            mask = (sub[column] == cat)
            v_ids.append(int(sub.loc[mask, id_col].dropna().nunique()))

        # place this split’s pair around its center offset
        center_offset = (-group_w if len(splits) == 3 else -group_w/2.0) + idx * group_w
        x_rows = x + center_offset - bar_w/2.0
        x_ids  = x + center_offset + bar_w/2.0

        color = SPLIT_TO_COLOR.get(split_name, "#888")
        # rows (solid)
        b_rows = ax.bar(x_rows, v_rows, bar_w, color=color, edgecolor="#333", linewidth=0.5)
        # ids  (hatched)
        b_ids  = ax.bar(x_ids,  v_ids,  bar_w, color=color, edgecolor="#333", linewidth=0.5, hatch="//", alpha=0.65)

        # labels: keep percentage/inside labels only for rows to avoid spam; id bars get a small count on top
        add_count_and_pct(ax, b_rows, v_rows, pct_base=max(int(split_sizes.get(split_name, 0)), 1), color_inside="white")
        add_count_label(ax, b_ids, v_ids)

        max_y = max(max_y, *(v_rows + v_ids))

        # legend (one entry per split color, plus one for "rows" and one for "IDs")
        if first_rows_handle is None:
            first_rows_handle = mpatches.Patch(facecolor="#999", edgecolor="#333", label="rows")
        if first_ids_handle is None:
            first_ids_handle = mpatches.Patch(facecolor="#999", edgecolor="#333", hatch="//", alpha=0.65, label="unique IDs")
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="#333", label=f"{split_name} (N={int(split_sizes.get(split_name, 0))})"))

    ax.set_title(f"{column} — Grouped by split (rows vs unique IDs)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticks(x, cats, rotation=15 if len(cats) > 5 else 0)
    ax.set_ylim(0, max_y * 1.55)

    # combine legends: split colors + pair meaning
    ax.legend(handles=legend_handles + [first_rows_handle, first_ids_handle],
              loc="upper right", frameon=True)
def plot_cat_simple_dual(ax, df: pd.DataFrame, column: str, id_col: str = "ID", color=BLUE, label=None, rotate_deg=None):
    if column not in df.columns:
        ax.axis("off"); ax.set_title(f"{column} (not found)", fontsize=TITLE_FONTSIZE); return

    # rows por categoría
    c_rows = df[column].dropna().value_counts()
    cats = c_rows.index.astype(str).tolist()
    vals_rows = c_rows.values.astype(int)

    # unique IDs por categoría
    vals_ids = []
    for cat in cats:
        mask = (df[column] == cat)
        vals_ids.append(int(df.loc[mask, id_col].dropna().nunique()))

    total_rows = int(sum(vals_rows))
    width = 0.35
    x = np.arange(len(cats))

    b_rows = ax.bar(x - width/2, vals_rows, width, color=color, edgecolor="#333", linewidth=0.5)
    b_ids  = ax.bar(x + width/2, vals_ids,  width, color=color, edgecolor="#333", linewidth=0.5, hatch="//", alpha=0.65)

    # labels: % sobre total solo para rows (para no saturar); IDs solo número
    add_count_and_pct(ax, b_rows, vals_rows, pct_base=max(total_rows,1), color_inside="white")
    add_count_label(ax, b_ids, vals_ids)

    ax.set_title(label or column, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(column)
    ax.set_ylabel("Count (rows / unique IDs)")
    rotation = rotate_deg if rotate_deg is not None else (15 if len(cats) > 5 else 0)
    ax.set_xticks(x, cats, rotation=rotation)
    for t in ax.get_xticklabels(): t.set_ha("right")

    max_y = max(list(vals_rows) + list(vals_ids)) if len(cats) else 1
    ax.set_ylim(0, max_y * 1.35)

    ax.legend(
        handles=[
            plt.Rectangle((0,0),1,1,color=color, label="rows"),
            plt.Rectangle((0,0),1,1,color=color, hatch="//", alpha=0.65, label="unique IDs"),
        ],
        loc="upper right", frameon=True
    )
def plot_groups_summary_dual(ax, df_seg: pd.DataFrame, id_counts: pd.Series):
    # imágenes por grupo (como antes)
    v = id_counts
    img_g1 = int(v[v <= 50].sum())
    img_g2 = int(v[(v >= 51) & (v <= 99)].sum())
    img_g3 = int(v[v >= 100].sum())

    # IDs por grupo
    g1_ids = (v[v <= 50]).index.size
    g2_ids = (v[(v >= 51) & (v <= 99)]).index.size
    g3_ids = (v[v >= 100]).index.size

    labels = ["≤50", "51-99", "≥100"]
    vals_rows = [img_g1, img_g2, img_g3]
    vals_ids  = [g1_ids, g2_ids, g3_ids]

    total_rows = int(sum(vals_rows))

    x = np.arange(len(labels))
    width = 0.35

    # colores por grupo como antes
    colors = [BLUE_G1, BLUE_G2, BLUE_G3]

    # barras de imágenes (sólidas)
    b_rows = ax.bar(x - width/2, vals_rows, width, color=colors, edgecolor="#333", linewidth=0.7)
    # barras de IDs (rayadas), mismo color + hatch
    b_ids  = ax.bar(x + width/2, vals_ids,  width, color=colors, edgecolor="#333", linewidth=0.7, hatch="//", alpha=0.65)

    add_count_and_pct(ax, b_rows, vals_rows, pct_base=max(total_rows,1), color_inside="white")
    add_count_label(ax, b_ids, vals_ids)

    ax.set_title("Images per ID — Group Summary (rows vs unique IDs)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Group")
    ax.set_ylabel("Sum(rows) / #IDs")
    ax.set_xticks(x, labels)
    max_y = max(vals_rows + vals_ids) if labels else 1
    ax.set_ylim(0, max_y * 1.22)

    # leyenda compacta
    ax.legend(
        handles=[
            plt.Rectangle((0,0),1,1,color="#999", label="rows (images)"),
            plt.Rectangle((0,0),1,1,color="#999", hatch="//", alpha=0.65, label="unique IDs"),
        ],
        loc="upper right", frameon=True
    )
def plot_cat_grouped_dual(ax, df_with_group: pd.DataFrame, column: str, id_col: str = "ID"):
    if column not in df_with_group.columns:
        ax.axis("off"); ax.set_title(f"{column} (not found)", fontsize=TITLE_FONTSIZE); return

    # separar por grupo
    parts = [
        ("G1 ≤50", BLUE_G1),
        ("G2 51-99", BLUE_G2),
        ("G3 ≥100", BLUE_G3),
    ]
    # categorías globales
    cats = sorted(df_with_group[column].dropna().unique().tolist())
    if not cats:
        ax.axis("off"); ax.set_title(f"{column} (empty)", fontsize=TITLE_FONTSIZE); return

    group_w = 0.25
    bar_w   = group_w * 0.45
    x = np.arange(len(cats))
    max_y = 1

    for idx, (gname, gcolor) in enumerate(parts):
        gdf = df_with_group[df_with_group["group"] == gname]
        N = int(len(gdf))

        # rows por categoría
        c_rows = gdf[column].dropna().value_counts()
        v_rows = [int(c_rows.get(cat, 0)) for cat in cats]

        # IDs únicos por categoría
        v_ids = []
        for cat in cats:
            mask = (gdf[column] == cat)
            v_ids.append(int(gdf.loc[mask, id_col].dropna().nunique()))

        center = (-group_w) + idx * group_w  # 3 grupos
        x_rows = x + center - bar_w/2.0
        x_ids  = x + center + bar_w/2.0

        b_rows = ax.bar(x_rows, v_rows, bar_w, color=gcolor, edgecolor="#333", linewidth=0.5, label=f"{gname} (N={N})" if idx==0 else None)
        b_ids  = ax.bar(x_ids,  v_ids,  bar_w, color=gcolor, edgecolor="#333", linewidth=0.5, hatch="//", alpha=0.65)

        add_count_and_pct(ax, b_rows, v_rows, pct_base=max(N,1), color_inside="white")
        add_count_label(ax, b_ids, v_ids)

        max_y = max(max_y, *(v_rows + v_ids))

    ax.set_title(f"{column} — Grouped (G1/G2/G3) rows vs unique IDs", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticks(x, cats, rotation=15 if len(cats) > 5 else 0)
    ax.set_ylim(0, max_y * 1.55)
    ax.legend(loc="upper right", frameon=True)
def plot_splits_summary_dual(ax, df: pd.DataFrame, id_col: str = "ID"):
    # rows por split
    counts = df["split"].value_counts()
    labels = [s for s in SPLIT_ORDER if s in counts.index]
    vals_rows = [int(counts.get(s, 0)) for s in labels]
    colors = [SPLIT_TO_COLOR[s] for s in labels]

    # IDs únicos por split
    vals_ids = []
    for s in labels:
        vals_ids.append(int(df.loc[df["split"] == s, id_col].dropna().nunique()))

    total_rows = int(sum(vals_rows))
    x = np.arange(len(labels))
    width = 0.35

    b_rows = ax.bar(x - width/2, vals_rows, width, color=colors, edgecolor="#333", linewidth=0.7)
    b_ids  = ax.bar(x + width/2, vals_ids,  width, color=colors, edgecolor="#333", linewidth=0.7, hatch="//", alpha=0.65)

    add_count_and_pct(ax, b_rows, vals_rows, pct_base=max(total_rows,1), color_inside="white")
    add_count_label(ax, b_ids, vals_ids)

    ax.set_title(f"Split Summary — rows vs unique IDs", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Split")
    ax.set_ylabel("Rows / #IDs")
    ax.set_xticks(x, labels)
    max_y = max(vals_rows + vals_ids) if labels else 1
    ax.set_ylim(0, max_y * 1.22)

    ax.legend(
        handles=[
            plt.Rectangle((0,0),1,1,color="#999", label="rows"),
            plt.Rectangle((0,0),1,1,color="#999", hatch="//", alpha=0.65, label="unique IDs"),
        ],
        loc="upper right", frameon=True
    )

# ============================
# MAIN (build three summaries)
# ============================
def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_root = Path(args.output_dir)
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # Segmento como en tu script original (IDs 1-35)
    if "ID" not in df.columns:
        raise ValueError("Se espera columna 'ID' (mayúsculas) para calcular imágenes por ID.")
    seg = df[df["ID"].between(1, 35)].copy()
    if seg.empty:
        raise ValueError("No hay filas para IDs 1-35 en el CSV.")

    # ---------- Datos base para GROUPS (G1/G2/G3) ----------
    seg_g, id_counts = attach_group_by_id_count(seg)

    # ---------- summary GENERAL ----------
    fig1 = plt.figure(figsize=(16, 20))
    gs1 = fig1.add_gridspec(5, 3, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.25)

    ax0 = fig1.add_subplot(gs1[0, :])
    plot_images_per_id(ax0, id_counts, color=BLUE, show_mean=True)

    ax1 = fig1.add_subplot(gs1[1, 0]); plot_hist(ax1, id_counts.values, color=BLUE)
    ax2 = fig1.add_subplot(gs1[1, 1]); plot_box(ax2, id_counts.values)

    cats_counts = {col: seg[col].dropna().value_counts() for col in CATEGORICAL_NAME_COLS if col in seg.columns}

    ax3  = fig1.add_subplot(gs1[1, 2]); plot_cat_simple_dual(ax3,  seg, "gender",         id_col="ID", color=BLUE, label="gender")
    ax4  = fig1.add_subplot(gs1[2, 0]); plot_cat_simple_dual(ax4,  seg, "colour",         id_col="ID", color=BLUE, label="colour")
    ax5  = fig1.add_subplot(gs1[2, 1]); plot_cat_simple_dual(ax5,  seg, "glasses/lenses", id_col="ID", color=BLUE, label="glasses/lenses")
    ax6  = fig1.add_subplot(gs1[2, 2]); plot_cat_simple_dual(ax6,  seg, "smoker",         id_col="ID", color=BLUE, label="smoker")
    ax7  = fig1.add_subplot(gs1[3, 0]); plot_cat_simple_dual(ax7,  seg, "phone_model",    id_col="ID", color=BLUE, label="phone_model", rotate_deg=10)
    ax8  = fig1.add_subplot(gs1[3, 1]); plot_cat_simple_dual(ax8,  seg, "light",          id_col="ID", color=BLUE, label="light")
    ax9  = fig1.add_subplot(gs1[3, 2]); plot_cat_simple_dual(ax9,  seg, "eye",            id_col="ID", color=BLUE, label="eye")
    ax10 = fig1.add_subplot(gs1[4, 0]); plot_cat_simple_dual(ax10, seg, "gaze",           id_col="ID", color=BLUE, label="gaze")

    fig1.add_subplot(gs1[4, 1]).axis("off")
    fig1.add_subplot(gs1[4, 2]).axis("off")

    fig1.suptitle("MOBIUS Summary — General (IDs 1-35)", fontsize=TITLE_FONTSIZE + 2, y=0.995)
    fig1.savefig(plots_dir / "summary_general.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ---------- SUMMARY PLOTS POR GRUPOS (G1/G2/G3) ----------
    fig2 = plt.figure(figsize=(27, 26))
    gs2 = fig2.add_gridspec(
        5, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        hspace=0.65,
        wspace=0.35
    )

    axg0 = fig2.add_subplot(gs2[0, :])
    plot_groups_summary_dual(axg0, seg_g, id_counts)

    axg1 = fig2.add_subplot(gs2[1, 0]); plot_cat_grouped_dual(axg1, seg_g, "gender")
    axg2 = fig2.add_subplot(gs2[1, 1]); plot_cat_grouped_dual(axg2, seg_g, "colour")
    axg3 = fig2.add_subplot(gs2[1, 2]); plot_cat_grouped_dual(axg3, seg_g, "glasses/lenses")
    axg4 = fig2.add_subplot(gs2[2, 0]); plot_cat_grouped_dual(axg4, seg_g, "smoker")
    axg5 = fig2.add_subplot(gs2[2, 1]); plot_cat_grouped_dual(axg5, seg_g, "phone_model")
    axg6 = fig2.add_subplot(gs2[2, 2]); plot_cat_grouped_dual(axg6, seg_g, "light")
    axg7 = fig2.add_subplot(gs2[3, 0]); plot_cat_grouped_dual(axg7, seg_g, "eye")
    axg8 = fig2.add_subplot(gs2[3, 1]); plot_cat_grouped_dual(axg8, seg_g, "gaze")
    fig2.add_subplot(gs2[3, 2]).axis("off")
    fig2.add_subplot(gs2[4, :]).axis("off")

    fig2.suptitle("MOBIUS Summary — Grouped (G1/G2/G3) (IDs 1-35)", fontsize=TITLE_FONTSIZE + 3, y=1.02)
    fig2.savefig(plots_dir / "summary_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ---------- NEW: SUMMARY PLOTS POR SPLIT (train/val/test) ----------
    if "split" not in seg.columns:
        raise ValueError("La columna 'split' no existe en el CSV. Asegúrate de haber agregado el split antes.")

    fig3 = plt.figure(figsize=(27, 26))
    gs3 = fig3.add_gridspec(
        5, 3,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
        hspace=0.65,
        wspace=0.35
    )

    axs0 = fig3.add_subplot(gs3[0, :])
    plot_splits_summary_dual(axs0, seg)

    # Categóricas agrupadas por split (3 barras train/val/test por categoría)
    # Categóricas agrupadas por split con doble barra (rows + unique IDs)
    axs1 = fig3.add_subplot(gs3[1, 0]); plot_cat_grouped_by_dual(axs1, seg, "gender", group_col="split", id_col="ID")
    axs2 = fig3.add_subplot(gs3[1, 1]); plot_cat_grouped_by_dual(axs2, seg, "colour", group_col="split", id_col="ID")
    axs3 = fig3.add_subplot(gs3[1, 2]); plot_cat_grouped_by_dual(axs3, seg, "glasses/lenses", group_col="split", id_col="ID")
    axs4 = fig3.add_subplot(gs3[2, 0]); plot_cat_grouped_by_dual(axs4, seg, "smoker", group_col="split", id_col="ID")
    axs5 = fig3.add_subplot(gs3[2, 1]); plot_cat_grouped_by_dual(axs5, seg, "phone_model", group_col="split", id_col="ID")
    axs6 = fig3.add_subplot(gs3[2, 2]); plot_cat_grouped_by_dual(axs6, seg, "light", group_col="split", id_col="ID")
    axs7 = fig3.add_subplot(gs3[3, 0]); plot_cat_grouped_by_dual(axs7, seg, "eye", group_col="split", id_col="ID")
    axs8 = fig3.add_subplot(gs3[3, 1]); plot_cat_grouped_by_dual(axs8, seg, "gaze", group_col="split", id_col="ID")

    fig3.add_subplot(gs3[3, 2]).axis("off")
    fig3.add_subplot(gs3[4, :]).axis("off")

    fig3.suptitle("MOBIUS Summary — Grouped by Split (train/val/test) (IDs 1-35)",
                  fontsize=TITLE_FONTSIZE + 3, y=1.02)
    fig3.savefig(plots_dir / "summary_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print("✅ summaries generados en:", plots_dir)
    print("   - summary_general.png")
    print("   - summary_groups.png")
    print("   - summary_splits.png")

def add_count_label(ax, bars, values, color_inside="white"):
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width()/2.0,
            rect.get_height(),
            f"{int(v)}",
            ha="center", va="bottom",
            fontsize=LABEL_FONTSIZE
        )
# ============================
# ARGS
# ============================
def parse_args():
    p = argparse.ArgumentParser(description="Generate MOBIUS summaries (general, grouped by size, grouped by split) using name columns only (IDs 1-35).")
    p.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()

if __name__ == "__main__":
    main()
