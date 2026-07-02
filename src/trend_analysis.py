# src/trend_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import pandas as pd
import numpy as np


#Optional dependency: only needed for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception:  
    plt = None
    mdates = None


@dataclass
class TrendConfig:
   

    clustered_csv: str = "data/processed/fashion_social_media_with_clusters_full.csv"
    #output files
    counts_csv: str = "data/processed/trends/instagram_cluster_counts.csv"
    share_csv: str = "data/processed/trends/instagram_cluster_share_pct.csv"
    plot_path: str = "data/processed/trends/clip_cluster_trend_evolution.png"

    
    
    labels_path: str = "data/processed/trends/cluster_labels.csv"

    #aggregation granularity
    freq: str = "M"  # month-end

    #filters
    platform_filter: str = "instagram"
    min_cluster_count: int = 20  #drop tiny clusters to reduce noise

    #plot options
    plot_start: Optional[str] = "2015-01-01"  #prevents weird early dates ruining the axis
    plot_end: Optional[str] = None  # e.g. "2016-12-31"
    rotate_xticks: int = 35
    figsize: Tuple[int, int] = (13, 6)
    dpi: int = 300

    #optional: reduce clutter by plotting only the top N clusters by total volume
    plot_top_n: Optional[int] = None  # e.g. 7


def _ensure_dir(path: Union[str, Path]) -> None:
    """
    ensures parent directory exists for a file path OR ensure directory exists if a directory is passed.
    """
    p = Path(path)
    #if it looks like a file (has a suffix), make its parent. Otherwise make the dir itself.
    target = p.parent if p.suffix else p
    target.mkdir(parents=True, exist_ok=True)


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    #tries best-effort parsing; invalid rows become NaT and are dropped later.
    return pd.to_datetime(series, errors="coerce", utc=False)


def compute_instagram_cluster_trends(
    cfg: Optional[TrendConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Returns:
      counts_df: rows = months, cols = cluster_ids (int)
      share_df:  rows = months, cols = cluster_ids (int), values in %
      meta:      dict with useful info
    """
    cfg = cfg or TrendConfig()

    clustered_path = Path(cfg.clustered_csv)
    if not clustered_path.exists():
        fallback = Path("data/processed/fashion_social_media_with_clusters.csv")
        if fallback.exists():
            clustered_path = fallback
        else:
            raise FileNotFoundError(
            f"Could not find clustered dataset at: {clustered_path} (or fallback {fallback}). "
            "Run clustering first (run_full_clusterings.py)."
        )

    df = pd.read_csv(clustered_path)

    required = {"platform", "cluster_id", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {clustered_path}: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    #filter to platform
    df = df[df["platform"].astype(str).str.lower() == cfg.platform_filter.lower()].copy()

    #parse timestamps and drop invalid
    df["timestamp"] = _coerce_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    #optional start/end filter
    if cfg.plot_start:
        start_dt = pd.to_datetime(cfg.plot_start)
        df = df[df["timestamp"] >= start_dt]
    if cfg.plot_end:
        end_dt = pd.to_datetime(cfg.plot_end)
        df = df[df["timestamp"] <= end_dt]

    #ensure cluster_id numeric-ish
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)

    #bucket timestamps to month
    #freq currently assumes monthly; to keep it robust, we still use period("M")
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    #drop tiny clusters (overall)
    overall_counts = df["cluster_id"].value_counts()
    keep_clusters = overall_counts[overall_counts >= cfg.min_cluster_count].index.tolist()
    df = df[df["cluster_id"].isin(keep_clusters)].copy()

    #pivot: counts per month per cluster
    counts_df = (
        df.groupby(["month", "cluster_id"])
        .size()
        .unstack("cluster_id", fill_value=0)
        .sort_index()
    )

    #share per month (percentage)
    row_sums = counts_df.sum(axis=1).replace(0, pd.NA)
    share_df = counts_df.div(row_sums, axis=0) * 100.0
    share_df = share_df.fillna(0.0)

    #save outputs
    _ensure_dir(cfg.counts_csv)
    _ensure_dir(cfg.share_csv)
    counts_df.to_csv(cfg.counts_csv, index=True)
    share_df.to_csv(cfg.share_csv, index=True)

    meta = {
        "n_rows_instagram": int(len(df)),
        "n_months": int(counts_df.shape[0]),
        "n_clusters_kept": int(counts_df.shape[1]),
        "clusters_kept": [int(c) for c in counts_df.columns.tolist()],
        "counts_csv": cfg.counts_csv,
        "share_csv": cfg.share_csv,
    }

    return counts_df, share_df, meta


def compute_instagram_cluster_trends_2(
    cfg: Optional[TrendConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backwards-compatible helper for code that expects only 2 return values:
      counts_df, share_df = compute_instagram_cluster_trends_2()
    """
    counts_df, share_df, _meta = compute_instagram_cluster_trends(cfg)
    return counts_df, share_df


def _load_label_map(cfg: TrendConfig) -> Optional[Dict[int, str]]:
    """
    Loads a mapping {cluster_id(int): label(str)} from cfg.labels_path.
    Returns None if file missing/empty/invalid.
    """
    labels_path = Path(cfg.labels_path)
    if not labels_path.exists():
        return None

    try:
        labels_df = pd.read_csv(labels_path)
    except Exception:
        return None

    if labels_df.empty:
        return None

    needed = {"cluster_id", "label"}
    if not needed.issubset(labels_df.columns):
        return None

    labels_df["cluster_id"] = pd.to_numeric(labels_df["cluster_id"], errors="coerce")
    labels_df = labels_df.dropna(subset=["cluster_id", "label"])
    if labels_df.empty:
        return None

    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    labels_df["label"] = labels_df["label"].astype(str).str.strip()

    #drop empty labels
    labels_df = labels_df[labels_df["label"].astype(str) != ""]
    if labels_df.empty:
        return None

    return dict(zip(labels_df["cluster_id"], labels_df["label"]))


def plot_instagram_cluster_share(
    share_df: pd.DataFrame,
    cfg: Optional[TrendConfig] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Plots the Instagram cluster share (%) over time.

    - Safe if labels CSV is missing/empty (falls back to 'Cluster X').
    - Uses cfg.plot_start/cfg.plot_end for readable window.
    - Optional cfg.plot_top_n reduces clutter by plotting only top N clusters by total volume.
    """
    cfg = cfg or TrendConfig()

    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is not installed. Install with: pip install matplotlib"
        )

    #ensure datetime index and sorted
    share_df = share_df.copy()
    if not isinstance(share_df.index, pd.DatetimeIndex):
        share_df.index = pd.to_datetime(share_df.index, errors="coerce")
    share_df = share_df[share_df.index.notna()].sort_index()

    #apply start/end filter
    if cfg.plot_start:
        start_dt = pd.to_datetime(cfg.plot_start)
        share_df = share_df[share_df.index >= start_dt]
    if cfg.plot_end:
        end_dt = pd.to_datetime(cfg.plot_end)
        share_df = share_df[share_df.index <= end_dt]

    if share_df.empty:
        raise ValueError("share_df is empty after date filtering; nothing to plot.")

    #optionally plot only top N clusters (by total share mass or by average share)
    if cfg.plot_top_n is not None:
        # Use total counts proxy: sum of shares across time (equivalent ranking)
        top_cols = share_df.sum(axis=0).sort_values(ascending=False).head(cfg.plot_top_n).index
        share_df = share_df[top_cols]

    #rename columns using labels (optional)
    label_map = _load_label_map(cfg)
    renamed = {}
    for c in share_df.columns:
        try:
            cid = int(c)
        except Exception:
            cid = c
        if label_map and isinstance(cid, int) and cid in label_map:
            renamed[c] = label_map[cid]
        else:
            renamed[c] = f"Cluster {cid}"
    share_df = share_df.rename(columns=renamed)

    #plot
    fig, ax = plt.subplots(figsize=cfg.figsize)

    for col in share_df.columns:
        ax.plot(share_df.index, share_df[col], marker="o", linewidth=2, label=col)

    ax.set_title("Temporal Evolution of CLIP-Discovered Fashion Archetypes (Instagram)", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of Posts (%)")
    ax.grid(alpha=0.3)

    #improve x-axis formatting
    if mdates is not None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(cfg.rotate_xticks)
        tick.set_ha("right")

    #puts legend outside to reduce clutter
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )

    fig.tight_layout()

    out_path = save_path or cfg.plot_path
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")
    return out_path

def compute_trend_scores(
    counts_df: pd.DataFrame,
    share_df: pd.DataFrame,
    cfg: Optional[TrendConfig] = None,
    save_csv: str = "data/processed/trends/instagram_trend_scores.csv",
    save_plot: str = "data/processed/trends/instagram_trend_scores_top.png",
    top_n: int = 6,
) -> pd.DataFrame:
    """
    Produces a simple, report-friendly trend scoring table per cluster.

    Inputs:
      - counts_df: monthly counts per cluster (rows=months, cols=cluster_id)
      - share_df:  monthly share% per cluster (rows=months, cols=cluster_id)

    Outputs:
      - CSV ranking clusters by "momentum_score" (higher = more emerging)
      - Optional bar chart of top emerging clusters
    """
    cfg = cfg or TrendConfig()

    # Ensure aligned indexes/columns
    counts_df = counts_df.copy()
    share_df = share_df.copy()
    counts_df.index = pd.to_datetime(counts_df.index, errors="coerce")
    share_df.index = pd.to_datetime(share_df.index, errors="coerce")
    counts_df = counts_df.sort_index()
    share_df = share_df.sort_index()

    common_idx = counts_df.index.intersection(share_df.index)
    common_cols = counts_df.columns.intersection(share_df.columns)
    counts_df = counts_df.loc[common_idx, common_cols]
    share_df = share_df.loc[common_idx, common_cols]

    #need at least 4 months for decent scoring
    if share_df.shape[0] < 4:
        raise ValueError(f"Not enough months to score trends. Have {share_df.shape[0]} months.")

  
    def _safe_pct_change(a: float, b: float) -> float:
        # percent change from a -> b (avoid divide-by-zero)
        if a is None or np.isnan(a) or a == 0:
            return np.nan
        return (b - a) / a * 100.0

    #label map (for readable output)
    label_map = _load_label_map(cfg) or {}

    rows = []
    for c in common_cols:
        cid = int(c)

        series = share_df[cid].astype(float).values
        counts = counts_df[cid].astype(float).values

        last = float(series[-1])
        prev = float(series[-2])
        delta_1m = last - prev
        pctchg_1m = _safe_pct_change(prev, last)

        #last 3 months average and slope (simple linear fit)
        last3 = series[-3:]
        x3 = np.arange(len(last3))
        slope_3m = float(np.polyfit(x3, last3, 1)[0])  # % points per month

        #overall slope across the whole window (more stable)
        x_all = np.arange(len(series))
        slope_all = float(np.polyfit(x_all, series, 1)[0])

        #volatility (std dev of monthly changes)
        diffs = np.diff(series)
        vol = float(np.std(diffs)) if len(diffs) else 0.0

        #support: recent counts (last 3 months mean) to downweight tiny clusters
        support_3m = float(np.mean(counts[-3:]))

        #momentum score:
        #prefers positive slope in recent months
        #penalises high volatility
        #lightly rewards higher recent share
        #lightly rewards having enough samples
        momentum = (
            2.0 * slope_3m
            + 0.3 * slope_all
            + 0.05 * last
            - 0.6 * vol
            + 0.002 * support_3m
        )

        rows.append(
            {
                "cluster_id": cid,
                "label": label_map.get(cid, f"Cluster {cid}"),
                "last_share_pct": round(last, 4),
                "delta_1m_pct_points": round(delta_1m, 4),
                "pct_change_1m": round(pctchg_1m, 4) if not np.isnan(pctchg_1m) else np.nan,
                "slope_3m_pct_points_per_month": round(slope_3m, 6),
                "slope_all_pct_points_per_month": round(slope_all, 6),
                "volatility_std_of_monthly_deltas": round(vol, 6),
                "support_mean_count_last3m": round(support_3m, 2),
                "momentum_score": round(float(momentum), 6),
            }
        )

    scores_df = pd.DataFrame(rows).sort_values("momentum_score", ascending=False).reset_index(drop=True)

    #save CSV
    _ensure_dir(save_csv)
    scores_df.to_csv(save_csv, index=False)
    print(f"Saved trend scores to: {save_csv}")

    #optional plot of top emerging clusters
    if plt is not None:
        top = scores_df.head(top_n).iloc[::-1]  #reverse for nicer horizontal ordering
        plt.figure(figsize=(12, 5))
        plt.barh(top["label"], top["momentum_score"])
        plt.title("Top Emerging CLIP Clusters (Momentum Score)")
        plt.xlabel("Momentum Score (higher = more emerging)")
        plt.tight_layout()
        _ensure_dir(save_plot)
        plt.savefig(save_plot, dpi=cfg.dpi)
        plt.close()
        print(f"Saved trend score plot to: {save_plot}")

    return scores_df


def run_trend_pipeline(cfg: Optional[TrendConfig] = None) -> Dict:
    """
    Convenience wrapper:
      - compute trend tables
      - save CSVs
      - save plot
      - compute + save trend scoring

    Returns meta dict with output paths.
    """
    cfg = cfg or TrendConfig()

    counts_df, share_df, meta = compute_instagram_cluster_trends(cfg)

    plot_path = plot_instagram_cluster_share(share_df, cfg)
    meta["plot_path"] = plot_path

    scores_df = compute_trend_scores(
        counts_df=counts_df,
        share_df=share_df,
        cfg=cfg,
    )
    meta["trend_scores_csv"] = "data/processed/trends/instagram_trend_scores.csv"
    meta["trend_scores_plot"] = "data/processed/trends/instagram_trend_scores_top.png"
    meta["top_emerging_labels"] = scores_df.head(5)["label"].tolist()

    return meta


plot_clip_cluster_trend_evolution = plot_instagram_cluster_share
plot_cluster_trend_evolution = plot_instagram_cluster_share
