from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class EmbeddingConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 32
    max_text_length: int = 77
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"
    cache_dir: str = "data/processed/embeddings_cache"
    timeout_s: int = 10
    user_agent: str = "cs3ip-degree-project/1.0"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    
    try:
        import torch  
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_unified_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    expected = {"platform", "image_path", "timestamp", "text"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unified CSV missing columns: {missing}. Found: {list(df.columns)}")
    return df


def split_platforms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ig = df[df["platform"].astype(str).str.lower() == "instagram"].copy()
    pin = df[df["platform"].astype(str).str.lower() == "pinterest"].copy()
    return ig, pin


def _safe_text(series: pd.Series) -> List[str]:
    return series.fillna("").astype(str).tolist()


def _safe_urls(series: pd.Series) -> List[str]:
    return series.fillna("").astype(str).tolist()


def _download_image_bytes(url: str, timeout_s: int, user_agent: str) -> Optional[bytes]:
    if not url or not isinstance(url, str):
        return None

   
    try:
        import requests  # type: ignore
    except Exception:
        raise ImportError("requests not installed. Install with: pip install requests")

    try:
        headers = {"User-Agent": user_agent}
        r = requests.get(url, timeout=timeout_s, headers=headers)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None


def get_clip_text_embeddings(
    texts: List[str],
    cfg: EmbeddingConfig,
) -> np.ndarray:
    """
    Returns embeddings with shape (N, D).
    """
    device = _resolve_device(cfg.device)

    try:
        import torch  
        from transformers import CLIPModel, CLIPProcessor  
    except Exception as e:
        raise ImportError(
            "Missing dependencies for CLIP text embeddings. Install with:\n"
            "  pip install torch transformers\n"
        ) from e

    model = CLIPModel.from_pretrained(cfg.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.model_name)

    model.eval()
    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size):
            batch = texts[i : i + cfg.batch_size]
            inputs = processor(
                text=batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_text_length,
                return_tensors="pt",
            ).to(device)

            feats = model.get_text_features(**inputs)
            
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                
                feats = feats.last_hidden_state.mean(dim=1)
            
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.detach().cpu().numpy())

    return np.vstack(all_embs) if all_embs else np.zeros((0, 512), dtype=np.float32)


def get_clip_image_embeddings_from_urls(
    image_urls: List[str],
    cfg: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads images from URLs and returns:
      - embeddings: (N_ok, D)
      - ok_mask: (N,) boolean mask indicating which URLs succeeded
    """
    device = _resolve_device(cfg.device)

    try:
        import torch  
        from PIL import Image  
        from transformers import CLIPModel, CLIPProcessor 
        from io import BytesIO
    except Exception as e:
        raise ImportError(
            "Missing dependencies for CLIP image embeddings. Install with:\n"
            "  pip install torch transformers pillow\n"
        ) from e

    model = CLIPModel.from_pretrained(cfg.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.model_name)
    model.eval()

    ok_mask = np.zeros(len(image_urls), dtype=bool)
    embs: List[np.ndarray] = []

    
    images: List[Image.Image] = []
    ok_indices: List[int] = []

    for idx, url in enumerate(image_urls):
        b = _download_image_bytes(url, cfg.timeout_s, cfg.user_agent)
        if b is None:
            continue
        try:
            img = Image.open(BytesIO(b)).convert("RGB")
        except Exception:
            continue

        ok_mask[idx] = True
        images.append(img)
        ok_indices.append(idx)

        #batch embed
        if len(images) >= cfg.batch_size:
            embs.append(_embed_image_batch(images, model, processor, device))
            images = []
            ok_indices = []

    #final batch
    if images:
        embs.append(_embed_image_batch(images, model, processor, device))

    if not embs:
        return np.zeros((0, 512), dtype=np.float32), ok_mask

    return np.vstack(embs), ok_mask


def _embed_image_batch(images, model, processor, device) -> np.ndarray:
    import torch  

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)

        #some versions return an output object rather than a tensor
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state.mean(dim=1)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy()


def cache_save(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def cache_load(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def build_embeddings_outputs(
    unified_csv_path: str | Path = "data/processed/fashion_social_media_unified.csv",
    cfg: Optional[EmbeddingConfig] = None,
) -> dict:
    """
    Creates embeddings for:
      - Instagram: CLIP text embeddings
      - Pinterest: CLIP image embeddings (URLs)
    Saves:
      - instagram_text_embeddings.npy
      - pinterest_image_embeddings.npy
      - pinterest_ok_mask.npy
    Returns paths and counts.
    """
    cfg = cfg or EmbeddingConfig()
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_unified_csv(unified_csv_path)
    ig, pin = split_platforms(df)

    #instagram text embeddings
    ig_texts = _safe_text(ig["text"])
    ig_out = cache_dir / "instagram_text_embeddings.npy"
    ig_emb = get_clip_text_embeddings(ig_texts, cfg)
    cache_save(ig_emb, ig_out)

    #pinterest image embeddings
    pin_urls = _safe_urls(pin["image_path"])
    pin_out = cache_dir / "pinterest_image_embeddings.npy"
    pin_mask_out = cache_dir / "pinterest_ok_mask.npy"
    pin_emb, ok_mask = get_clip_image_embeddings_from_urls(pin_urls, cfg)
    cache_save(pin_emb, pin_out)
    cache_save(ok_mask.astype(np.int8), pin_mask_out)

    return {
        "instagram_rows": int(len(ig)),
        "pinterest_rows": int(len(pin)),
        "instagram_embeddings_path": str(ig_out),
        "pinterest_embeddings_path": str(pin_out),
        "pinterest_ok_mask_path": str(pin_mask_out),
        "instagram_embeddings_shape": tuple(ig_emb.shape),
        "pinterest_embeddings_shape": tuple(pin_emb.shape),
    }