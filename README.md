# Multimodal Fashion Trend Analysis using Vision-Language Models

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

An end-to-end machine learning system for analysing multimodal social media content to discover and forecast emerging fashion trends using **vision-language models, unsupervised learning, temporal analysis, and interactive visualisation**.

The project integrates image and text data from Instagram and Pinterest, leveraging **OpenAI CLIP** embeddings to learn shared semantic representations before identifying latent fashion archetypes through clustering and analysing their evolution over time.

> 🎓 Developed as a First Class undergraduate dissertation in Computer Science at the University of Reading.

---

# 🚀 Live Interactive Dashboard

**Explore the deployed Streamlit application**

🔗 https://fashion-trend-discovery-dashboard.streamlit.app/

---

# 📌 Project Highlights

- Analysed **25,000+ multimodal social media records**
- Implemented **OpenAI CLIP** for joint image-text embeddings
- Built a complete **multimodal machine learning pipeline**
- Applied **K-Means clustering** for fashion archetype discovery
- Performed **temporal trend analysis** across social media activity
- Compared **Linear Regression** against **Naïve forecasting baselines**
- Developed an interactive **Streamlit dashboard** for real-time exploration
- Selected for demonstration at the **University of Reading Open Day**

---

# 🖼 Dashboard Preview

> Replace these placeholders with screenshots from your dashboard.

| Home | Trend Analysis |
|------|----------------|
| ![](assets/dashboard_home.png) | ![](assets/trend_analysis.png) |

| Forecasting | Cluster Explorer |
|-------------|------------------|
| ![](assets/forecasting.png) | ![](assets/cluster_explorer.png) |

---

# 🎯 Project Motivation

Social media platforms have become major drivers of fashion trends, producing vast amounts of multimodal data through images, captions and user interactions.

Traditional trend forecasting relies heavily on expert judgement and historical sales data. This project investigates whether **vision-language models** and **machine learning** can automatically discover emerging fashion patterns directly from large-scale social media content.

The objective is not simply to classify images, but to understand how visual and textual information combine to reveal meaningful fashion archetypes and their evolution over time.

---

# 🏗 Machine Learning Pipeline

```
Instagram Captions
            │
Pinterest Images
            │
────────────────────────
     Data Preprocessing
────────────────────────
            │
     OpenAI CLIP Encoder
            │
 Shared Image–Text Embeddings
            │
      K-Means Clustering
            │
 Temporal Trend Analysis
            │
 Forecasting (Linear vs Naïve)
            │
 Interactive Streamlit Dashboard
```

---

# ⚙️ Methodology

## 1. Data Collection

- Instagram captions (STL Dataset)
- Pinterest fashion images

---

## 2. Data Preprocessing

- Caption cleaning
- Timestamp normalisation
- Image validation
- Dataset integration

---

## 3. Representation Learning

Images and text are encoded into a shared semantic embedding space using **OpenAI CLIP**, enabling multimodal similarity learning and downstream clustering.

---

## 4. Unsupervised Learning

Fashion archetypes are discovered using **K-Means clustering**, with silhouette analysis used to determine the optimal number of clusters.

---

## 5. Temporal Analysis

Monthly cluster frequencies are analysed to measure trend momentum and identify emerging or declining fashion styles.

---

## 6. Forecasting

Trend trajectories are forecast using:

- Linear Regression
- Naïve Baseline

Performance is evaluated using:

- MAE
- MAPE

---

## 7. Interactive Dashboard

The Streamlit dashboard enables users to:

- Explore discovered fashion clusters
- Compare forecasting models
- Visualise trend evolution
- Inspect representative image examples

---

# 🛠 Technologies

## Programming

- Python

## Machine Learning

- OpenAI CLIP
- PyTorch
- Scikit-learn
- Transformers

## Data Science

- Pandas
- NumPy

## Visualisation

- Plotly
- Streamlit

## Development

- Git
- Jupyter Notebook
- VS Code

---

# 📂 Repository Structure

```
.
├── data/
├── pages/
├── src/
├── assets/
├── scripts/
├── Home.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Key Results

- Successfully generated multimodal embeddings representing both visual and textual fashion content.
- Identified interpretable fashion archetypes using unsupervised clustering.
- Demonstrated that multimodal social media data can reveal evolving fashion trends.
- Found that Naïve forecasting consistently outperformed Linear Regression for highly volatile social media trends.
- Developed an interactive dashboard enabling intuitive exploration of discovered clusters and trend evolution.

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/Pranaviiiii/multimodal-fashion-trend-analysis.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run Home.py
```

---

# 📈 Future Improvements

Potential future extensions include:

- Fine-tuning CLIP for fashion-specific representations
- Integration of TikTok and additional social media platforms
- Transformer-based forecasting models
- Graph Neural Networks for social trend propagation
- Real-time streaming data pipelines
- Edge deployment optimisation for mobile inference
- Improved explainability using attention visualisation

---

# 👩‍💻 Author

**Pranavi Rawal**

First Class BSc (Hons) Computer Science  
University of Reading

LinkedIn: https://linkedin.com/in/pranavi-rawal

GitHub: https://github.com/Pranaviiiii

---

# 📄 License

This repository is released for academic and portfolio purposes.

Datasets remain subject to their respective licences.
