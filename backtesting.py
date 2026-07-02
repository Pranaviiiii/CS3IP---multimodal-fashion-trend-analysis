from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    share_csv: str = "data/processed/trends/instagram_cluster_share_pct.csv"
    labels_csv: str = "data/processed/trends/cluster_labels.csv"
    out_csv: str = "data/processed/trends/forecast_backtest_metrics.csv"
    min_history: int = 5  #need 5 for test


def _load_label_map(labels_csv: str) -> Dict[int, str]:
    path = Path(labels_csv)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "cluster_id" not in df.columns or "label" not in df.columns:
        return {}

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id", "label"])
    df["cluster_id"] = df["cluster_id"].astype(int)

    return dict(zip(df["cluster_id"], df["label"]))


def run_forecast_backtest(cfg: Optional[BacktestConfig] = None) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()

    share_path = Path(cfg.share_csv)
    if not share_path.exists():
        raise FileNotFoundError(
            f"Could not find trend share CSV at {share_path}. Run trend pipeline first."
        )

    share_df = pd.read_csv(share_path, index_col=0)
    share_df.index = pd.to_datetime(share_df.index, errors="coerce")
    share_df = share_df.sort_index()

    label_map = _load_label_map(cfg.labels_csv)

    rows: List[Dict] = []

    for col in share_df.columns:
        cid = int(col)
        y = share_df[col].astype(float).values

        if len(y) < cfg.min_history:
            continue

        #train on all except final point
        y_train = y[:-1]
        y_true = float(y[-1])

        #linear trend model
        x_train = np.arange(len(y_train))
        slope, intercept = np.polyfit(x_train, y_train, 1)

        next_x = len(y_train)
        y_pred_linear = float(slope * next_x + intercept)
        y_pred_linear = max(y_pred_linear, 0.0)

        #naive persistence baseline: predict next value = last observed training value
        y_pred_naive = float(y_train[-1])
        y_pred_naive = max(y_pred_naive, 0.0)

        mae_linear = abs(y_true - y_pred_linear)
        mae_naive = abs(y_true - y_pred_naive)

        mape_linear = (
            abs((y_true - y_pred_linear) / y_true) * 100 if y_true != 0 else np.nan
        )
        mape_naive = (
            abs((y_true - y_pred_naive) / y_true) * 100 if y_true != 0 else np.nan
        )

        better_model = (
            "linear"
            if mae_linear < mae_naive
            else "naive"
            if mae_naive < mae_linear
            else "tie"
        )

        rows.append(
            {
                "cluster_id": cid,
                "label": label_map.get(cid, f"Cluster {cid}"),
                "actual_last_share_pct": round(y_true, 4),
                "predicted_last_share_pct_linear": round(y_pred_linear, 4),
                "predicted_last_share_pct_naive": round(y_pred_naive, 4),
                "mae_linear": round(mae_linear, 4),
                "mae_naive": round(mae_naive, 4),
                "mape_pct_linear": round(float(mape_linear), 4)
                if not np.isnan(mape_linear)
                else np.nan,
                "mape_pct_naive": round(float(mape_naive), 4)
                if not np.isnan(mape_naive)
                else np.nan,
                "better_model": better_model,
                "mae_improvement_vs_naive": round(mae_naive - mae_linear, 4),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["mae_linear", "mae_naive"])
    Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.out_csv, index=False)

    print(f"Saved backtest metrics to: {cfg.out_csv}")
    return out_df