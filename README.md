# Machine Learning-Based Analysis of Social Media Content for Fashion Trend Discovery

## Overview

This project investigates whether multimodal social media data can be used to identify and forecast fashion trends. The system combines Instagram caption data and Pinterest image data using modern machine learning techniques, including multimodal representation learning, unsupervised clustering, temporal trend analysis, and forecasting.

The project was developed as a final-year Computer Science dissertation focused on understanding how social media platforms influence modern fashion trend formation.

link for interactive dashboard ->> https://fashion-trend-discovery-dashboard.streamlit.app/
---

## Project Objectives

The main objectives of the project were:

* Analyse large-scale social media fashion data
* Combine textual and visual fashion content into a shared feature space
* Discover latent fashion archetypes using unsupervised learning
* Track the evolution of fashion trends over time
* Evaluate whether fashion trends can be reliably forecasted
* Build an interactive dashboard for trend exploration and forecasting visualisation

---

## Technologies Used

### Programming Language

* Python 3

### Machine Learning & Data Science Libraries

* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* transformers
* torch
* open_clip

### Dashboard & Visualisation

* Streamlit
* Plotly

### Other Tools

* Jupyter Notebook
* VS Code
* GitHub

---

## Dataset Sources

### Instagram Dataset

* STL Dataset (Shop The Look)
* Source: [https://github.com/kang205/STL-Dataset](https://github.com/kang205/STL-Dataset)

### Pinterest Dataset

* Pinterest Fashion Dataset
* Source: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7AW6F](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7AW6F)

---

## Methodology

# Machine Learning-Based Analysis of Social Media Content for Fashion Trend Discovery

## Overview

This project investigates whether multimodal social media data can be used to identify and forecast fashion trends. The system combines Instagram caption data and Pinterest image data using modern machine learning techniques, including multimodal representation learning, unsupervised clustering, temporal trend analysis, and forecasting.

The project was developed as a final-year Computer Science dissertation focused on understanding how social media platforms influence modern fashion trend formation.

link for interactive dashboard ->> https://fashion-trend-discovery-dashboard.streamlit.app/
---

## Project Objectives

The main objectives of the project were:

* Analyse large-scale social media fashion data
* Combine textual and visual fashion content into a shared feature space
* Discover latent fashion archetypes using unsupervised learning
* Track the evolution of fashion trends over time
* Evaluate whether fashion trends can be reliably forecasted
* Build an interactive dashboard for trend exploration and forecasting visualisation

---

## Technologies Used

### Programming Language

* Python 3

### Machine Learning & Data Science Libraries

* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* transformers
* torch
* open_clip

### Dashboard & Visualisation

* Streamlit
* Plotly

### Other Tools

* Jupyter Notebook
* VS Code
* GitHub

---

## Dataset Sources

### Instagram Dataset

* STL Dataset (Shop The Look)
* Source: [https://github.com/kang205/STL-Dataset](https://github.com/kang205/STL-Dataset)

### Pinterest Dataset

* Pinterest Fashion Dataset
* Source: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7AW6F](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K7AW6F)

---

## Methodology

The project follows a modular machine learning pipeline:

### 1. Data Collection

* Instagram captions used as textual fashion data
* Pinterest images used as visual fashion data

### 2. Data Preprocessing

* Caption cleaning and normalisation
* Timestamp filtering and temporal formatting
* Image validation and preprocessing
* Unified multimodal dataset construction

### 3. Feature Representation

* CLIP (Contrastive Language-Image Pretraining) used for embedding generation
* Text and images encoded into a shared embedding space

### 4. Clustering

* K-Means clustering applied to multimodal embeddings
* Silhouette analysis used for K selection
* Final clustering performed with K = 8

### 5. Temporal Trend Analysis

* Monthly aggregation of cluster distributions
* Trend momentum calculation
* Analysis of trend evolution over time

### 6. Forecasting

* Linear Regression forecasting model
* Naive baseline forecasting model
* Forecast evaluation using MAE and MAPE

### 7. Dashboard Development

* Interactive Streamlit dashboard
* Trend analysis visualisations
* Forecasting visualisations
* Cluster exploration interface

---

## Repository Structure

```text
DEGREE_PROJECT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── schema.md
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── pages/
│   ├── 1_Trend_Analysis.py
│   ├── 2_Forecasting.py
│   └── 3_Trend_Explorer.py
│
├── src/
│   ├── backtesting.py
│   ├── clustering.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── exemplars.py
The project follows a modular machine learning pipeline:

### 1. Data Collection

* Instagram captions used as textual fashion data
* Pinterest images used as visual fashion data

### 2. Data Preprocessing

* Caption cleaning and normalisation
* Timestamp filtering and temporal formatting
* Image validation and preprocessing
* Unified multimodal dataset construction

### 3. Feature Representation

* CLIP (Contrastive Language-Image Pretraining) used for embedding generation
* Text and images encoded into a shared embedding space

### 4. Clustering

* K-Means clustering applied to multimodal embeddings
* Silhouette analysis used for K selection
* Final clustering performed with K = 8

### 5. Temporal Trend Analysis

* Monthly aggregation of cluster distributions
* Trend momentum calculation
* Analysis of trend evolution over time

### 6. Forecasting

* Linear Regression forecasting model
* Naive baseline forecasting model
* Forecast evaluation using MAE and MAPE

### 7. Dashboard Development

* Interactive Streamlit dashboard
* Trend analysis visualisations
* Forecasting visualisations
* Cluster exploration interface

---

## Repository Structure

```text
DEGREE_PROJECT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── schema.md
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── pages/
│   ├── 1_Trend_Analysis.py
│   ├── 2_Forecasting.py
│   └── 3_Trend_Explorer.py
│
├── src/
│   ├── backtesting.py
│   ├── clustering.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── exemplars.py
│   ├── forecasting.py
│   └── trend_analysis.py
│
├── assets/
│   ├── cluster_image_grid.png
│   ├── dataset_composition.png
│   ├── final_poster_graph.png
│   └── modality_example.png
│
├── scripts/
│   ├── plot_k_sweep.py
│   ├── rebuild_pinterest_exemplars.py
│   ├── run_backtest.py
│   ├── run_full_clusterings.py
│   ├── run_full_embeddings.py
│   ├── run_k_sweep.py
│   ├── run_pipeline.py
│   └── test_clip_sample.py
│
├── app.py
├── Home.py
├── presentation.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Dashboard Features

The Streamlit dashboard includes:

* Interactive trend analysis visualisations
* Cluster evolution over time
* Forecast comparison charts
* Cluster explorer with exemplar content
* Fashion trend momentum rankings
* Forecasting model evaluation

---

## Key Findings

* Multimodal social media data can successfully reveal meaningful fashion patterns
* CLIP embeddings effectively capture semantic relationships between text and images
* K-Means clustering identifies interpretable fashion archetypes
* Fashion trends on social media are highly dynamic and volatile
* Naive forecasting baselines outperform linear regression in many cases
* Social media is effective for trend detection but difficult for reliable forecasting

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
python -m streamlit run home.py
```

---

│   ├── forecasting.py
│   └── trend_analysis.py
│
├── assets/
│   ├── cluster_image_grid.png
│   ├── dataset_composition.png
│   ├── final_poster_graph.png
│   └── modality_example.png
│
├── scripts/
│   ├── plot_k_sweep.py
│   ├── rebuild_pinterest_exemplars.py
│   ├── run_backtest.py
│   ├── run_full_clusterings.py
│   ├── run_full_embeddings.py
│   ├── run_k_sweep.py
│   ├── run_pipeline.py
│   └── test_clip_sample.py
│
├── app.py
├── Home.py
├── presentation.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Dashboard Features

The Streamlit dashboard includes:

* Interactive trend analysis visualisations
* Cluster evolution over time
* Forecast comparison charts
* Cluster explorer with exemplar content
* Fashion trend momentum rankings
* Forecasting model evaluation

---

## Key Findings

* Multimodal social media data can successfully reveal meaningful fashion patterns
* CLIP embeddings effectively capture semantic relationships between text and images
* K-Means clustering identifies interpretable fashion archetypes
* Fashion trends on social media are highly dynamic and volatile
* Naive forecasting baselines outperform linear regression in many cases
* Social media is effective for trend detection but difficult for reliable forecasting

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
python -m streamlit run home.py
```

---

## Future Improvements

Potential future work includes:

* Integrating TikTok and additional social media platforms
* Using Instagram image data directly
* Extending the temporal range of the dataset
* Applying advanced forecasting models such as ARIMA and LSTM
* Implementing bias mitigation techniques for platform-driven datasets
* Improving quantitative cluster validation

---

## Author

Pranavi Rawal
Final Year Computer Science Project
University of Reading

---

## License

This project was developed for academic purposes.


