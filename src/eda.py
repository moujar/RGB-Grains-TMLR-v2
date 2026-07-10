#!/usr/bin/env python3
"""
Advanced Exploratory Data Analysis (EDA) for the RGB-Grains datasets.

The datasets under ``data/`` are collections of per-grain crops stored as
``.npz`` files with two arrays:

    x                     (252, 252, 3) int16   3 spectral bands [22, 53, 89]
    means_over_spectralon (3,)          float64  per-channel spectralon reference

Grains are segmented on a black (exactly-zero) background, so foreground masks
and morphology can be recovered directly from ``x``.

This module builds a rich, decoupled EDA (no torch dependency, headless-safe):

  * A metadata table parsed purely from filenames (fast, no image load):
        dataset · kind · label · microplot/bac · year · illumination · datetime
  * Image-level statistics sampled per class (streamed, memory-bounded):
        active-pixel area · fill fraction · per-channel foreground reflectance
        · brightness · spectralon reference
  * A suite of figures + a Markdown report + summary CSVs written to ``--out``.

Filename / label conventions mirror ``src/dataset.py``:
  * pure stand   grain166_var1-x75y20_7000_us_2x_2021-10-19T160916_corr.npz
  * mixed stand  grain130_x38y23-mix19_8000_us_2x_2020-12-03T160859_corr.npz
  * SCOOP bacs   grain19_R22-scoop-bac74-3_8000_us_2x_2022-08-26T072529_corr.npz

Usage
-----
    python src/eda.py                          # scans ./data, writes ./eda_outputs
    python src/eda.py --data-dir data --out eda_outputs --max-per-class 400
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# ── CONSTANTS (kept in sync with src/dataset.py) ──────────────────────────────
CH_SCALE = np.array([1567.0, 8316.0, 18126.0])  # per-band scaling → reflectance
SPECTRAL_BANDS = [22, 53, 89]                    # band indices (blue-ish, green, NIR)
CH_LABELS = [f"band {b}" for b in SPECTRAL_BANDS]

# variety index → name (alphabetical, 0..7), matching dataset.py
VARIETY_NAMES = [
    "ACCROC", "AUBUSSON", "BAGOU", "BELEPI",
    "BERGAMO", "BOREGAR", "EXPERT", "KALAHARI",
]

# SCOOP bac → variety (from README §5.2 bis). Bacs not listed are flagged.
SCOOP_BAC_TO_VARIETY = {
    14: "EL4X-199", 18: "EL4X-199", 43: "EL4X-199",
    17: "EL4X-35", 71: "EL4X-35", 74: "EL4X-35",
    41: "EL4X-482", 42: "EL4X-482", 50: "EL4X-482",
    7: "GQ4X-83", 57: "GQ4X-83", 92: "GQ4X-83",
}

# A qualitative, colour-blind-friendly palette reused across figures
PALETTE = sns.color_palette("colorblind", 12)


# ── FILENAME PARSING ──────────────────────────────────────────────────────────
_RE_VAR = re.compile(r"var(\d{1,2})")
_RE_MIX = re.compile(r"mix(\d{1,2})")
_RE_BAC = re.compile(r"bac(\d+)-(\d+)")
_RE_MICROPLOT = re.compile(r"x\d{2}y\d{2}")
_RE_DATETIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6})")
_RE_ILLUM = re.compile(r"_(\d+)_us_2x_")


def _load_mix_dict(csv_path: str = "perfomix_mixtures.csv") -> dict[str, list[str]]:
    """Return {mixNN: [variety names]} from the mixtures table (see dataset.py)."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, sep="\t")
    out: dict[str, list[str]] = {}
    for mix_name, group in df.groupby("mix"):
        canon = f"mix{int(str(mix_name)[3:]):02d}"
        vars = sorted(group["var"].tolist())
        out[canon] = vars
        out[str(mix_name)] = vars
    return out


def parse_filename(path: str, mix_dict: dict) -> dict:
    """Extract all metadata derivable from a grain ``.npz`` filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.basename(os.path.dirname(path))

    row: dict = {
        "path": path,
        "filename": stem,
        "dataset": folder,
        "kind": None,
        "label": None,        # canonical class id (string)
        "label_name": None,   # human-readable variety / mixture
        "n_components": 1,     # varieties in the (possibly mixed) sample
        "microplot": None,
        "replicate": None,
        "year": None,
        "illumination": None,
        "datetime": None,
    }

    m = _RE_ILLUM.search(stem)
    if m:
        row["illumination"] = int(m.group(1))

    m = _RE_DATETIME.search(stem)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H%M%S")
            row["datetime"] = dt
            row["year"] = dt.year
        except ValueError:
            pass

    mp = _RE_MICROPLOT.search(stem)
    if mp:
        row["microplot"] = mp.group(0)

    bac = _RE_BAC.search(stem)
    var = _RE_VAR.search(stem)
    mix = _RE_MIX.search(stem)

    if bac:  # SCOOP
        bac_id, rep = int(bac.group(1)), int(bac.group(2))
        row["kind"] = "bac"
        row["label"] = f"bac{bac_id}"
        row["label_name"] = SCOOP_BAC_TO_VARIETY.get(bac_id, f"bac{bac_id}?")
        row["microplot"] = f"bac{bac_id}"
        row["replicate"] = rep
    elif var:  # pure stand
        vi = int(var.group(1)) - 1
        row["kind"] = "pure"
        row["label"] = f"var{vi + 1}"
        row["label_name"] = VARIETY_NAMES[vi] if 0 <= vi < len(VARIETY_NAMES) else f"var{vi + 1}"
    elif mix:  # mixed stand
        canon = f"mix{int(mix.group(1)):02d}"
        row["kind"] = "mix"
        row["label"] = canon
        comps = mix_dict.get(canon) or mix_dict.get(f"mix{int(mix.group(1))}")
        if comps:
            row["label_name"] = "+".join(comps)
            row["n_components"] = len(comps)
        else:
            row["label_name"] = canon
    else:
        row["kind"] = "unknown"

    return row


def build_metadata(data_dir: str, mix_dict: dict) -> pd.DataFrame:
    """Scan every ``.npz`` under ``data_dir`` and parse its filename metadata."""
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    rows = [parse_filename(f, mix_dict) for f in files]
    df = pd.DataFrame(rows)
    return df


# ── IMAGE-LEVEL STATISTICS (streamed, sampled) ────────────────────────────────
def image_stats(path: str) -> dict | None:
    """Compute lightweight per-grain statistics from one ``.npz`` (streamed)."""
    try:
        d = np.load(path)
        x = d["x"].astype(np.float32)
    except Exception:
        return None
    fg = (x > 0).any(axis=2)
    area = int(fg.sum())
    out = {"area": area, "fill_frac": area / (x.shape[0] * x.shape[1])}
    if area == 0:
        for i in range(3):
            out[f"refl{i}"] = 0.0
        out["brightness"] = 0.0
    else:
        refl = x[fg] / CH_SCALE  # (area, 3) normalized reflectance
        m = refl.mean(axis=0)
        for i in range(3):
            out[f"refl{i}"] = float(m[i])
        out["brightness"] = float(m.mean())
    if "means_over_spectralon" in d:
        sp = d["means_over_spectralon"]
        for i in range(3):
            out[f"spectralon{i}"] = float(sp[i])
    return out


def sample_image_stats(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Attach image statistics for up to ``max_per_class`` grains per (dataset,label)."""
    rng = np.random.default_rng(seed)
    picks = []
    for (_, _), g in df.groupby(["dataset", "label"], dropna=False):
        idx = g.index.to_numpy()
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        picks.extend(idx.tolist())
    sub = df.loc[picks].copy()
    stats = [image_stats(p) for p in sub["path"]]
    stats_df = pd.DataFrame([s for s in stats], index=sub.index)
    out = sub.join(stats_df)
    return out.dropna(subset=["area"])


# ── RENDERING HELPERS ─────────────────────────────────────────────────────────
def render_rgb(path: str, crop: int | None = 176) -> np.ndarray | None:
    """Load a grain and render it the way the model sees it: x/CH_SCALE, clipped."""
    try:
        x = np.load(path)["x"].astype(np.float32)
    except Exception:
        return None
    img = np.clip(x / CH_SCALE, 0.0, 1.0)
    if crop and img.shape[0] > crop and img.shape[1] > crop:
        t = (img.shape[0] - crop) // 2
        l = (img.shape[1] - crop) // 2
        img = img[t:t + crop, l:l + crop]
    return img


def _savefig(fig, out_dir: Path, name: str):
    fig.tight_layout()
    p = out_dir / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")
    return p


# ── FIGURES ───────────────────────────────────────────────────────────────────
def fig_inventory(df: pd.DataFrame, out: Path):
    counts = df.groupby(["dataset", "kind"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="barh", stacked=True, ax=ax, color=PALETTE)
    ax.set_xlabel("number of grain crops")
    ax.set_ylabel("")
    ax.set_title("Dataset inventory — grains per folder, by kind")
    for c in ax.containers:
        ax.bar_label(c, label_type="center", fmt=lambda v: f"{int(v)}" if v else "", fontsize=9)
    ax.legend(title="kind", loc="lower right")
    _savefig(fig, out, "01_inventory.png")


def fig_class_balance(df: pd.DataFrame, out: Path):
    kinds = [("pure", "Pure perfomix varieties"), ("bac", "SCOOP bacs → variety")]
    present = [(k, t) for k, t in kinds if (df["kind"] == k).any()]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(7 * len(present), 6), squeeze=False)
    for ax, (k, title) in zip(axes[0], present):
        sub = df[df["kind"] == k]
        col = "label_name"
        order = sub[col].value_counts().index
        sns.countplot(data=sub, y=col, order=order, ax=ax, palette="viridis", hue=col, legend=False)
        ax.set_title(title)
        ax.set_xlabel("count")
        ax.set_ylabel("")
        ax.bar_label(ax.containers[0], fontsize=10, padding=2)
    fig.suptitle("Class balance", y=1.02)
    _savefig(fig, out, "02_class_balance.png")


def fig_microplot_heatmap(df: pd.DataFrame, out: Path):
    for kind, tag in [("pure", "pure"), ("bac", "SCOOP")]:
        sub = df[(df["kind"] == kind) & df["microplot"].notna()]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="label_name", columns="microplot",
                                values="path", aggfunc="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1] + 4), max(5, 0.5 * pivot.shape[0] + 2)))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="rocket_r", ax=ax, cbar_kws={"label": "grains"})
        ax.set_title(f"Grains per microplot × class ({tag})\n"
                     "splits are by microplot → each column is a train/test unit")
        ax.set_xlabel("microplot")
        ax.set_ylabel("")
        _savefig(fig, out, f"03_microplot_heatmap_{tag}.png")


def fig_illumination(df: pd.DataFrame, out: Path):
    sub = df[df["illumination"].notna()]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = sub.groupby(["dataset", "illumination"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=ax, color=PALETTE)
    ax.set_title("Illumination / integration setting per dataset")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="setting (µs)")
    _savefig(fig, out, "04_illumination.png")


def fig_timeline(df: pd.DataFrame, out: Path):
    sub = df[df["datetime"].notna()].copy()
    if sub.empty:
        return
    sub["date"] = sub["datetime"].dt.date
    ct = sub.groupby(["date", "dataset"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 5))
    ct.plot(kind="area", stacked=True, ax=ax, alpha=0.8, color=PALETTE)
    ax.set_title("Acquisition timeline — grains captured per day")
    ax.set_ylabel("grains / day")
    ax.set_xlabel("")
    ax.legend(title="dataset", fontsize=8)
    _savefig(fig, out, "05_timeline.png")


def fig_area(stats: pd.DataFrame, out: Path):
    if stats.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # per-class breakdown only for the pure-label kinds (mixtures → 50+ rows, unreadable)
    pc = stats[stats["kind"].isin(["pure", "bac"])]
    if pc.empty:
        pc = stats
    order = pc.groupby("label_name")["area"].median().sort_values().index
    sns.violinplot(data=pc, x="area", y="label_name", order=order, ax=axes[0],
                   density_norm="width", cut=0, hue="label_name", legend=False, palette="mako")
    axes[0].set_title("Active-pixel area per class (pure & SCOOP)")
    axes[0].set_xlabel("foreground pixels (area)")
    axes[0].set_ylabel("")

    sns.histplot(data=stats, x="area", hue="kind", element="step", stat="density",
                 common_norm=False, ax=axes[1], palette="Set2")
    # data-cleaning guidance from README §5.5 (BACS: >0.01 refl area < 110000 → tiny)
    axes[1].axvline(stats["area"].quantile(0.02), color="crimson", ls="--",
                    label="2nd percentile (candidate small-grain cutoff)")
    axes[1].set_title("Area distribution (all sampled grains)")
    axes[1].set_xlabel("foreground pixels (area)")
    axes[1].legend(fontsize=9)
    _savefig(fig, out, "06_grain_area.png")


def fig_spectral(stats: pd.DataFrame, out: Path):
    """Per-class spectral signature, one figure per pure-label dataset kind."""
    if stats.empty:
        return
    for kind, tag in [("pure", "pure"), ("bac", "SCOOP")]:
        sub = stats[stats["kind"] == kind]
        if sub.empty:
            continue
        long = sub.melt(id_vars=["label_name"],
                        value_vars=["refl0", "refl1", "refl2"],
                        var_name="band", value_name="reflectance")
        long["band"] = long["band"].map(
            {"refl0": CH_LABELS[0], "refl1": CH_LABELS[1], "refl2": CH_LABELS[2]})
        order = sorted(sub["label_name"].unique())
        fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(order) + 3), 6.5))
        sns.violinplot(data=long, x="label_name", y="reflectance", hue="band",
                       order=order, ax=ax, density_norm="width", cut=0, palette="Set1")
        ax.set_title(f"Mean foreground reflectance per class & spectral band ({tag})\n"
                     "spectral separability preview — the signal the classifier keys on")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="band", loc="best")
        _savefig(fig, out, f"07_spectral_signature_{tag}.png")


def fig_channel_scatter(stats: pd.DataFrame, out: Path):
    """Per-grain 2D reflectance scatter (band53 vs band89), coloured by class."""
    for kind in ["pure", "bac"]:
        sub = stats[stats["kind"] == kind]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.scatterplot(data=sub, x="refl1", y="refl2", hue="label_name",
                        alpha=0.45, s=22, ax=ax, palette="tab10", edgecolor="none")
        ax.set_xlabel(f"reflectance {CH_LABELS[1]}")
        ax.set_ylabel(f"reflectance {CH_LABELS[2]}")
        ax.set_title(f"Foreground reflectance by grain ({kind})")
        ax.legend(title="class", fontsize=8, loc="best")
        _savefig(fig, out, f"08_reflectance_scatter_{kind}.png")


def fig_mixtures(df: pd.DataFrame, mix_dict: dict, out: Path):
    sub = df[df["kind"] == "mix"]
    if sub.empty:
        return
    # variety frequency across mixture *samples*
    var_counter: Counter = Counter()
    for _, r in sub.iterrows():
        comps = mix_dict.get(r["label"]) or []
        for v in comps:
            var_counter[v] += 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    if var_counter:
        vs = pd.Series(var_counter).sort_values(ascending=True)
        axes[0].barh(vs.index, vs.values, color=PALETTE)
        axes[0].bar_label(axes[0].containers[0], fontsize=9)
        axes[0].set_title("Variety occurrence across mixed-stand grains")
        axes[0].set_xlabel("grains containing this variety")
    card = sub["n_components"].value_counts().sort_index()
    axes[1].bar(card.index.astype(str), card.values, color=sns.color_palette("flare", len(card)))
    axes[1].bar_label(axes[1].containers[0], fontsize=10)
    axes[1].set_title("Mixture cardinality (varieties per mixed sample)")
    axes[1].set_xlabel("number of varieties in mixture")
    axes[1].set_ylabel("grains")
    _savefig(fig, out, "09_mixtures.png")


def fig_spectralon(stats: pd.DataFrame, out: Path):
    if "spectralon2" not in stats.columns or stats["spectralon2"].isna().all():
        return
    sub = stats.dropna(subset=["spectralon2"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    long = sub.melt(id_vars=["dataset"], value_vars=["spectralon0", "spectralon1", "spectralon2"],
                    var_name="band", value_name="spectralon")
    long["band"] = long["band"].map({"spectralon0": CH_LABELS[0], "spectralon1": CH_LABELS[1], "spectralon2": CH_LABELS[2]})
    sns.boxplot(data=long, x="band", y="spectralon", hue="dataset", ax=axes[0])
    axes[0].set_title("Spectralon reference per band & dataset")
    axes[0].legend(fontsize=7, title="dataset")
    if "datetime" in sub.columns and sub["datetime"].notna().any():
        sub2 = sub.dropna(subset=["datetime"])
        sns.scatterplot(data=sub2, x="datetime", y="spectralon2", hue="dataset",
                        alpha=0.5, s=18, ax=axes[1], legend=False)
        axes[1].set_title(f"Spectralon ({CH_LABELS[2]}) drift over acquisition time")
        axes[1].tick_params(axis="x", rotation=30)
    _savefig(fig, out, "10_spectralon.png")


def fig_montage(df: pd.DataFrame, out: Path, kind: str, n_per: int, seed: int,
                max_rows: int = 12):
    """Per-class montage. When a kind has many classes (mixtures), fall back to a
    flat random grid so grains stay large enough to read."""
    sub = df[df["kind"] == kind]
    if sub.empty:
        return
    rng = np.random.default_rng(seed)
    classes = sorted(sub["label_name"].dropna().unique())

    if len(classes) > max_rows:  # flat grid, grains kept large
        paths = sub["path"].to_numpy()
        pick = rng.choice(paths, min(max_rows * n_per, len(paths)), replace=False)
        ncol = n_per
        nrow = int(np.ceil(len(pick) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.7 * nrow), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < len(pick):
                img = render_rgb(pick[i])
                if img is not None:
                    ax.imshow(img)
        fig.suptitle(f"Example grains — {kind} ({len(classes)} classes, random sample; "
                     "rendered as x / CH_SCALE)", y=1.002)
        _savefig(fig, out, f"11_montage_{kind}.png")
        return

    fig, axes = plt.subplots(len(classes), n_per,
                             figsize=(1.7 * n_per, 1.7 * len(classes)), squeeze=False)
    for r, cls in enumerate(classes):
        paths = sub[sub["label_name"] == cls]["path"].to_numpy()
        pick = rng.choice(paths, min(n_per, len(paths)), replace=False)
        for c in range(n_per):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(pick):
                img = render_rgb(pick[c])
                if img is not None:
                    ax.imshow(img)
            if c == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=10)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
    fig.suptitle(f"Example grains — {kind} (rendered as x / CH_SCALE, model view)", y=1.005)
    _savefig(fig, out, f"11_montage_{kind}.png")


# ── REPORT ────────────────────────────────────────────────────────────────────
def write_report(df: pd.DataFrame, stats: pd.DataFrame, out: Path, mix_dict: dict):
    lines: list[str] = []
    A = lines.append
    A("# RGB-Grains — Exploratory Data Analysis\n")
    A(f"Generated from `{out.parent}` · **{len(df):,} grain crops** across "
      f"**{df['dataset'].nunique()} folders**.\n")

    A("## 1. Inventory\n")
    inv = df.groupby(["dataset", "kind"]).size().rename("grains").reset_index()
    A(inv.to_markdown(index=False))
    A("")

    A("## 2. Classes\n")
    for kind in ["pure", "bac", "mix"]:
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue
        # for SCOOP the semantic class is the variety (label_name), not the bac (label)
        vc = sub["label_name"].value_counts()
        note = " (bacs → varieties)" if kind == "bac" else ""
        A(f"**{kind}** — {sub['label_name'].nunique()} classes{note}, "
          f"{len(sub):,} grains, imbalance ratio "
          f"{vc.max() / max(vc.min(), 1):.2f} (max/min class).\n")

    A("## 3. Split-relevant structure (microplots)\n")
    for kind, tag in [("pure", "pure"), ("bac", "SCOOP")]:
        sub = df[(df["kind"] == kind) & df["microplot"].notna()]
        if sub.empty:
            continue
        per = sub.groupby("label_name")["microplot"].nunique()
        A(f"- **{tag}**: {sub['microplot'].nunique()} microplots; "
          f"{per.min()}–{per.max()} microplots per class "
          f"(median {int(per.median())}). Splits are by microplot, so this bounds "
          f"the number of achievable cross-validation folds.")
    A("")

    if not stats.empty:
        A("## 4. Morphology & radiometry (sampled)\n")
        A(f"Sampled **{len(stats):,} grains** for pixel-level stats.\n")
        area = stats["area"]
        A(f"- Active-pixel area: median **{int(area.median())}**, "
          f"IQR [{int(area.quantile(.25))}, {int(area.quantile(.75))}], "
          f"range [{int(area.min())}, {int(area.max())}].")
        tiny = int((area < area.quantile(0.02)).sum())
        A(f"- **{tiny}** sampled grains fall below the 2nd-percentile area "
          f"(candidate dust/half-grain outliers → README §5.5 cleaning).")
        A("- Mean foreground reflectance per band (all sampled):")
        for i, lab in enumerate(CH_LABELS):
            col = stats[f"refl{i}"]
            A(f"  - {lab}: mean {col.mean():.3f} ± {col.std():.3f}")
        A("")

    miss = df[df["label_name"].astype(str).str.endswith("?")]
    if not miss.empty:
        A("## 5. Data caveats\n")
        A(f"- **{len(miss)}** grains have an unmapped label "
          f"(e.g. SCOOP bac not in the README mapping): "
          f"{sorted(miss['label'].unique())}.")
        A("")

    report = out / "eda_report.md"
    report.write_text("\n".join(lines))
    print(f"  [saved] {report}")

    df.drop(columns=["path"]).to_csv(out / "metadata.csv", index=False)
    if not stats.empty:
        stats.drop(columns=["path"]).to_csv(out / "image_stats_sample.csv", index=False)
    print(f"  [saved] {out / 'metadata.csv'}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Advanced EDA for RGB-Grains datasets")
    ap.add_argument("--data-dir", default="data", help="root folder of *_processed datasets")
    ap.add_argument("--out", default="eda_outputs", help="output directory")
    ap.add_argument("--mix-csv", default="perfomix_mixtures.csv")
    ap.add_argument("--max-per-class", type=int, default=400,
                    help="grains sampled per class for pixel-level stats/scatter")
    ap.add_argument("--montage-n", type=int, default=8, help="example grains per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    plt.switch_backend("Agg")  # headless for CLI runs; notebooks keep their own backend
    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Scanning {args.data_dir} ...")
    mix_dict = _load_mix_dict(args.mix_csv)
    df = build_metadata(args.data_dir, mix_dict)
    if df.empty:
        print("No .npz files found — nothing to do.")
        return
    print(f"[*] Parsed {len(df):,} grain crops across {df['dataset'].nunique()} folders.")

    print(f"[*] Sampling image stats (≤{args.max_per_class}/class) ...")
    stats = sample_image_stats(df, args.max_per_class, args.seed)
    print(f"    → {len(stats):,} grains measured.")

    print("[*] Building figures ...")
    figure_fns = [
        lambda: fig_inventory(df, fig_dir),
        lambda: fig_class_balance(df, fig_dir),
        lambda: fig_microplot_heatmap(df, fig_dir),
        lambda: fig_illumination(df, fig_dir),
        lambda: fig_timeline(df, fig_dir),
        lambda: fig_area(stats, fig_dir),
        lambda: fig_spectral(stats, fig_dir),
        lambda: fig_channel_scatter(stats, fig_dir),
        lambda: fig_mixtures(df, mix_dict, fig_dir),
        lambda: fig_spectralon(stats, fig_dir),
        lambda: fig_montage(df, fig_dir, "pure", args.montage_n, args.seed),
        lambda: fig_montage(df, fig_dir, "bac", args.montage_n, args.seed),
        lambda: fig_montage(df, fig_dir, "mix", args.montage_n, args.seed),
    ]
    for fn in figure_fns:
        try:
            fn()
        except Exception as e:  # keep going if a single figure fails
            print(f"  [warn] figure failed: {e}")

    print("[*] Writing report ...")
    write_report(df, stats, out, mix_dict)
    print(f"[✓] EDA complete → {out.resolve()}")


if __name__ == "__main__":
    main()
