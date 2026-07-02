# Multimodal Fashion Trend Analysis using Vision-Language Models

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Multimodal-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

A machine learning project that analyses multimodal social media content to identify and forecast fashion trends. The system combines image and text data using OpenAI CLIP embeddings, applies unsupervised clustering to discover fashion archetypes, and analyses how these trends change over time through an interactive Streamlit dashboard.

This project was completed as my undergraduate dissertation for the BSc Computer Science programme at the University of Reading and was selected for demonstration at the University's Open Day.

---

## Live Demo

The interactive dashboard is available here:

**https://fashion-trend-discovery-dashboard.streamlit.app/**

---

## Key Features

- Analyses more than **25,000** Instagram and Pinterest records.
- Generates multimodal image-text embeddings using **OpenAI CLIP**.
- Identifies fashion archetypes using **K-Means clustering**.
- Tracks how fashion clusters change over time.
- Compares Linear Regression against a Naïve forecasting baseline.
- Provides an interactive Streamlit dashboard for exploring results.

---

## Project Pipeline

```
Instagram Captions
          │
Pinterest Images
          │
Data Preprocessing
          │
OpenAI CLIP Embeddings
          │
K-Means Clustering
          │
Temporal Trend Analysis
          │
Forecasting
          │
Streamlit Dashboard
```

---

## Repository Structure

```
.
├── assets/
├── notebooks/
├── pages/
├── scripts/
├── src/
├── Home.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Method

The project follows the workflow below.

### 1. Data Collection

Two public datasets were used:

- Pinterest fashion images
- Instagram captions

### 2. Data Preparation

Before modelling, the datasets were cleaned and standardised by:

- validating image files
- cleaning text
- formatting timestamps
- combining both datasets

### 3. Feature Extraction

OpenAI CLIP was used to generate shared image-text embeddings, allowing images and captions to be represented within the same embedding space.

### 4. Clustering

K-Means clustering was applied to discover groups of visually and semantically similar fashion styles. Silhouette analysis was used to determine the optimal number of clusters.

### 5. Trend Analysis

Monthly cluster frequencies were analysed to identify emerging, stable and declining fashion trends.

### 6. Forecasting

Two forecasting approaches were evaluated:

- Linear Regression
- Naïve Forecast

Performance was compared using MAE and MAPE.

---

## Results

The project demonstrates that multimodal embeddings can successfully identify meaningful fashion archetypes from social media data.

Some of the main findings include:

- CLIP embeddings produced coherent image-text representations.
- K-Means successfully grouped similar fashion styles.
- Temporal analysis highlighted changing popularity across fashion clusters.
- The Naïve Forecast outperformed Linear Regression on this dataset due to the volatility of social media trends.

---

## Dataset

The original datasets are **not included** in this repository.

This is intentional because:

- the datasets significantly increase the repository size;
- the original datasets are distributed under their own licences and are therefore not redistributed here.

The repository contains all source code required to reproduce the project. Users wishing to run the full pipeline should download the original datasets and place them in the expected directory structure.

---

## Technologies

**Programming**

- Python

**Machine Learning**

- OpenAI CLIP
- Scikit-learn

**Libraries**

- Pandas
- NumPy
- Plotly
- Streamlit

**Development**

- Jupyter Notebook
- VS Code
- Git

---

## Installation

Clone the repository

```bash
git clone https://github.com/Pranaviiiii/multimodal-fashion-trend-analysis.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run Home.py
```

---

## Future Work

Possible extensions include:

- fine-tuning CLIP for fashion-specific representations;
- incorporating additional social media platforms such as TikTok;
- evaluating transformer-based forecasting models;
- developing a real-time data collection pipeline;
- improving explainability of cluster assignments.

---

## Author

**Pranavi Rawal**

BSc (Hons) Computer Science — First Class Honours  
University of Reading

LinkedIn: https://linkedin.com/in/pranavi-rawal

GitHub: https://github.com/Pranaviiiii

---

## Licence

This repository is intended for academic and portfolio purposes.

The datasets used during development remain subject to their original licences.
