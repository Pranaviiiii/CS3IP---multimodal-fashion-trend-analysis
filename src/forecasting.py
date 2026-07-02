from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ForecastConfig:
    share_csv: str = "data/processed/trends/instagram_cluster_share_pct.csv"
    labels_csv: str = "data/processed/trends/cluster_labels.csv"
    out_csv: str = "data/processed/trends/cluster_forecasts.csv"
    horizon: int = 3  #forecast next 3 months
    min_history: int = 4  #minimum months needed to fit trend


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


def forecast_cluster_trends(cfg: Optional[ForecastConfig] = None) -> pd.DataFrame:
    cfg = cfg or ForecastConfig()

    share_path = Path(cfg.share_csv)
    if not share_path.exists():
        raise FileNotFoundError(
            f"Could not find trend share CSV at {share_path}. Run trend pipeline first."
        )

    share_df = pd.read_csv(share_path, index_col=0)
    share_df.index = pd.to_datetime(share_df.index, errors="coerce")
    share_df = share_df.sort_index()

    label_map = _load_label_map(cfg.labels_csv)

    results: List[Dict] = []

    for col in share_df.columns:
        cid = int(col)
        y = share_df[col].astype(float).values

        if len(y) < cfg.min_history:
            continue

        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)

        last_date = share_df.index[-1]
        last_observed = float(y[-1])

        for step in range(1, cfg.horizon + 1):
            future_x = len(y) - 1 + step
            pred_linear = float(slope * future_x + intercept)
            pred_linear = max(pred_linear, 0.0)

            #naive persistence baseline:
            #future value assumed equal to the most recent observed value
            pred_naive = max(last_observed, 0.0)

            future_date = last_date + pd.DateOffset(months=step)

            results.append(
                {
                    "cluster_id": cid,
                    "label": label_map.get(cid, f"Cluster {cid}"),
                    "forecast_month": future_date.strftime("%Y-%m-%d"),
                    "forecast_step": step,
                    "predicted_share_pct": round(pred_linear, 4),
                    "predicted_share_pct_naive": round(pred_naive, 4),
                    "slope": round(float(slope), 6),
                    "intercept": round(float(intercept), 6),
                    "last_observed_share_pct": round(last_observed, 4),
                }
            )

    out_df = pd.DataFrame(results)
    Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.out_csv, index=False)

    print(f"Saved forecasts to: {cfg.out_csv}")
    return out_df
