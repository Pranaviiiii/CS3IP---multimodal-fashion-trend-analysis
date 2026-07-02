# src/exemplars.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class ExemplarConfig:
    #prefer the latest clustering output (k=12) but fall back if needed
    unified_with_clusters_csv: str = "data/processed/fashion_social_media_with_clusters_full.csv"
    cache_dir: str = "data/processed/embeddings_cache"
    out_dir: str = "data/processed/exemplars"
    top_n: int = 5


def _resolve_clusters_csv(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)

    #fallback to older filename if present
    fallback = Path("data/processed/fashion_social_media_with_clusters.csv")
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        f"Could not find clustered CSV at '{path}' or fallback '{fallback}'. "
        "Run clustering first (run_full_clusterings.py)."
    )


def _load_embeddings(cache_dir: str) -> Dict[str, np.ndarray]:
    cache = Path(cache_dir)
    ig = np.load(cache / "instagram_text_embeddings.npy", allow_pickle=False)
    pin = np.load(cache / "pinterest_image_embeddings.npy", allow_pickle=False)
    pin_mask = np.load(cache / "pinterest_ok_mask.npy", allow_pickle=False).astype(bool)
    return {"ig": ig, "pin": pin, "pin_mask": pin_mask}


def _l2_normalise(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def _cosine_similarity(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between each row of A and vector b.
    Assumes A and b are L2-normalised.
    """
    return A @ b


def build_cluster_exemplars(
    cfg: ExemplarConfig = ExemplarConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds exemplar tables for Pinterest (images) and Instagram (captions).
    Saves two CSV files:
      - data/processed/exemplars/pinterest_exemplars.csv
      - data/processed/exemplars/instagram_exemplars.csv
    """
    clusters_csv = _resolve_clusters_csv(cfg.unified_with_clusters_csv)
    df = pd.read_csv(clusters_csv)

    #basic checks
    required = {"platform", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {clusters_csv}: {sorted(missing)}")

    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)

    embs = _load_embeddings(cfg.cache_dir)
    ig_emb = embs["ig"]
    pin_emb = embs["pin"]
    pin_mask = embs["pin_mask"]

    #normalise embeddings (safe even if already normalised)
    ig_emb = _l2_normalise(ig_emb)
    pin_emb = _l2_normalise(pin_emb)

    #split rows
    ig_df = df[df["platform"].astype(str).str.lower() == "instagram"].copy()
    pin_df = df[df["platform"].astype(str).str.lower() == "pinterest"].copy()

    #instagram labels align with ig_emb directly
    ig_labels = ig_df["cluster_id"].to_numpy()

    #pinterest: only rows with successful downloads have valid embeddings
    pin_df_ok = pin_df.loc[pin_mask].copy()
    pin_labels_ok = pin_df_ok["cluster_id"].to_numpy()

    
    pin_emb_ok = pin_emb[pin_mask]

    clusters = sorted(df["cluster_id"].unique())

    pinterest_rows = []
    instagram_rows = []

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in clusters:
        ig_idx = np.where(ig_labels == c)[0]
        pin_idx = np.where(pin_labels_ok == c)[0]

        vectors = []
        if ig_idx.size > 0:
            vectors.append(ig_emb[ig_idx])
        if pin_idx.size > 0:
            vectors.append(pin_emb_ok[pin_idx])

        if not vectors:
            continue

        centroid = np.vstack(vectors).mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        #pinterest exemplars
        if pin_idx.size > 0:
            sims = _cosine_similarity(pin_emb_ok[pin_idx], centroid)
            top_local = np.argsort(-sims)[: cfg.top_n]

            for rank, j in enumerate(top_local, start=1):
                row = pin_df_ok.iloc[pin_idx[j]]
                pinterest_rows.append(
                    {
                        "cluster_id": int(c),
                        "rank": int(rank),
                        "image_path": row.get("image_path", ""),
                        "text": row.get("text", ""),
                    }
                )

        #instagram exemplars
        if ig_idx.size > 0:
            sims = _cosine_similarity(ig_emb[ig_idx], centroid)
            top_local = np.argsort(-sims)[: cfg.top_n]

            for rank, j in enumerate(top_local, start=1):
                row = ig_df.iloc[ig_idx[j]]
                instagram_rows.append(
                    {
                        "cluster_id": int(c),
                        "rank": int(rank),
                        "text": row.get("text", ""),
                        "timestamp": row.get("timestamp", ""),
                    }
                )

    pin_out = pd.DataFrame(pinterest_rows)
    ig_out = pd.DataFrame(instagram_rows)

    pin_out_path = out_dir / "pinterest_exemplars.csv"
    ig_out_path = out_dir / "instagram_exemplars.csv"

    pin_out.to_csv(pin_out_path, index=False)
    ig_out.to_csv(ig_out_path, index=False)

    print(f"Saved: {pin_out_path}")
    print(f"Saved: {ig_out_path}")

    return pin_out, ig_out
