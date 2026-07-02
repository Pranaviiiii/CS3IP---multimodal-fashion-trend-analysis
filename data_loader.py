"""
Data ingestion module for CS3IP project:
'Machine Learning-Based Analysis of Instagram and Pinterest Content
for Fashion Trend Discovery'

Design rationale:
- Uses public datasets only (ethical, GDPR-safe)
- Unified schema supports multimodal, unsupervised analysis
- Image is primary modality; text is optional support

Aligned with literature:
- Exploratory trend discovery
- No predefined fashion labels
"""



import pandas as pd


def map_instagram_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)
    mapped["platform"] = "instagram"
    mapped["image_path"] = df["ImgURL"].astype(str)
    mapped["timestamp"] = pd.to_datetime(df["CreationTime"], unit="s", errors="coerce")
    mapped["text"] = df["Caption"].fillna("").astype(str) + " " + df["Hashtags"].fillna("").astype(str)
    return mapped.reset_index(drop=True)


def map_pinterest_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.DataFrame(index=df.index)
    mapped["platform"] = "pinterest"
    mapped["image_path"] = df["image_url"].fillna("").astype(str)
    mapped["timestamp"] = pd.NaT
    mapped["text"] = df["image_description"].fillna("").astype(str)
    return mapped.reset_index(drop=True)


