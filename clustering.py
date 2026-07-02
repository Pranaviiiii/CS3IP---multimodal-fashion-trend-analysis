from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


import numpy as np
import pandas as pd


@dataclass
class ClusteringConfig:
    k: int = 8
    random_state: int = 42
    cache_dir: str = "data/processed/embeddings_cache"
    out_dir: str = "data/processed"
    unified_csv_path: str = "data/processed/fashion_social_media_unified.csv"


def _load_embeddings(cache_dir: str) -> Dict[str, np.ndarray]:
    cache = Path(cache_dir)
    ig = np.load(cache / "instagram_text_embeddings.npy", allow_pickle=False)
    pin = np.load(cache / "pinterest_image_embeddings.npy", allow_pickle=False)
    pin_mask = np.load(cache / "pinterest_ok_mask.npy", allow_pickle=False).astype(bool)
    return {"ig": ig, "pin": pin, "pin_mask": pin_mask}


def _load_unified(unified_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(unified_csv_path)


def cluster_joint_embeddings(cfg: Optional[ClusteringConfig] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Clusters:
      - all Instagram text embeddings
      - Pinterest image embeddings (only those that downloaded successfully)

    Writes cluster_id back into:
      data/processed/fashion_social_media_with_clusters.csv
    """
    cfg = cfg or ClusteringConfig()
    embs = _load_embeddings(cfg.cache_dir)

    ig_emb = embs["ig"]
    pin_emb = embs["pin"]
    pin_mask = embs["pin_mask"]

    #combine for clustering
    X = np.vstack([ig_emb, pin_emb])

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as e:
        raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn") from e

    km = KMeans(n_clusters=cfg.k, random_state=cfg.random_state, n_init="auto")
    labels = km.fit_predict(X)

    sil = None
    if X.shape[0] >= cfg.k + 2:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

    #map labels back to rows
    df = _load_unified(cfg.unified_csv_path)

    ig_idx = df["platform"].astype(str).str.lower() == "instagram"
    pin_idx = df["platform"].astype(str).str.lower() == "pinterest"

    df["cluster_id"] = np.nan

    #instagram labels (first block)
    df.loc[ig_idx, "cluster_id"] = labels[: ig_emb.shape[0]]

    #pinterest labels (second block) — only for successful image downloads
    pin_labels = labels[ig_emb.shape[0] :]

    pin_rows = df.loc[pin_idx].copy()
    pin_rows.loc[pin_mask, "cluster_id"] = pin_labels
    df.loc[pin_idx, "cluster_id"] = pin_rows["cluster_id"].values

    out_path = Path(cfg.out_dir) / "fashion_social_media_with_clusters_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta = {
        "k": cfg.k,
        "silhouette": sil,
        "output_csv": str(out_path),
        "instagram_emb_shape": tuple(ig_emb.shape),
        "pinterest_emb_shape": tuple(pin_emb.shape),
        "pinterest_success_rate": float(pin_mask.mean()) if pin_mask.size else 0.0,
    }
    return df, meta

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute multiple unsupervised cluster quality metrics.
    """
    metrics = {}

    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:
        metrics["silhouette"] = float(silhouette_score(X, labels))
    else:
        metrics["silhouette"] = float("nan")

    metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))

    counts = np.bincount(labels)
    metrics["min_cluster_size"] = int(counts.min())
    metrics["max_cluster_size"] = int(counts.max())
    metrics["n_clusters"] = int(len(unique_labels))

    return metrics