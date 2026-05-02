# ⚽ FIFA Match Outcome Prediction

> Predicting international football match results using historical data, PySpark ETL pipelines, and XGBoost — with live match batch processing support.

---

## 📌 Project Overview

This project builds a machine learning pipeline to predict the outcomes of international football matches. It leverages over 150 years of match history, performs large-scale data processing with PySpark, and trains an XGBoost classifier to forecast match results. Ongoing match data is ingested and processed as batch streams for real-time-ready predictions.

---

## 🗂️ Dataset

| Source | Details |
|--------|---------|
| **Kaggle** | [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| **GitHub** | [martj42/international_results](https://github.com/martj42/international_results) |

The dataset contains results of international football matches from **1872 to present**, including home team, away team, scores, tournament type, city, country, and neutral venue flag. The GitHub source is actively maintained with ongoing match results, which feeds directly into the batch processing pipeline.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python |
| ETL / Big Data | PySpark |
| Data Storage | Delta Tables |
| Data Processing | Pandas |
| ML Model | XGBoost |
| Feature Engineering | Scikit-learn (sklearn) |

---

## 🔄 Pipeline Architecture

```
Raw Data (CSV / Streaming)
        │
        ▼
┌───────────────────┐
│   ETL (PySpark)   │  ← Extract, Transform, Load
│   Delta Tables    │  ← Versioned storage
└───────────────────┘
        │
        ▼
┌───────────────────────┐
│   Pre-processing      │  ← Data Cleaning
│   (Null handling,     │
│    type casting,      │
│    deduplication)     │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Feature Engineering  │  ← Historical win rates, form,
│                       │    goal stats, ELO-style scores
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│   Train / Test Split  │  ← sklearn train_test_split
│   XGBoost Classifier  │  ← Gradient boosting model
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Batch Processing     │  ← Ongoing/live match data
│  (Streaming Ingestion)│    processed as micro-batches
└───────────────────────┘
        │
        ▼
   Match Predictions
```

---

## ⚙️ Modules

### 1. ETL — `etl/`
- Ingests raw CSV data using **PySpark**
- Cleans and loads data into **Delta Tables** for reliable, versioned access
- Supports both historical batch loads and ongoing match batch ingestion

### 2. Pre-processing — `preprocessing/`
- Handles missing values and nulls
- Type casting and schema enforcement
- Removes duplicate records
- Normalizes team names and tournament categories

### 3. Feature Engineering — `feature_engineering/`
- Generates the training and testing feature dataset
- Key engineered features include:
  - Home / Away win rates (last N matches)
  - Goal scoring and conceding averages
  - Head-to-head historical record
  - Tournament importance weighting
  - Venue type (neutral / home ground)
  - Recent form (rolling window stats)

### 4. Model Training — `model/`
- Train/test split using **sklearn**
- **XGBoost** classifier for match outcome prediction (Win / Loss)
- Hyperparameter tuning and cross-validation
- Model evaluation: Accuracy, F1-score, Confusion Matrix

### 5. Batch Streaming — `streaming/`
- Ongoing match data is collected and ingested
- Processed as **batch processing** via PySpark structured streaming
- Predictions generated for live fixture data

---

## 🚀 Getting Started

### Prerequisites

```bash
Python > 3.9
Apache Spark = 4.1.1
OpenJDK >= 17 (required for PySpark)
```

### Installation

```bash
git clone [https://github.com/your-username/fifa-prediction.git](https://github.com/ManikandanRJSM/fifa_prediction_analysis.git)
cd fifa_prediction_analysis

virtualenv venv
source venv/bin/activate
pip install pyspark==4.1.1 delta-spark==4.1.0 pandas==2.3.3 requests==2.25.1 python-dotenv==1.2.2 scikit-learn==1.7.2 xgboost==3.2.0 joblib==1.5.3 matplotlib==3.10.9 seaborn==0.13.2 numpy==2.2.6
```

### Requirements

```
pyspark
pandas
scikit-learn
xgboost
delta-spark
virtualenv
numpy
```
---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | _TBD after training_ |
| F1-Score (macro) | _TBD_ |
| Train/Test Split | _TBD_ |

> Results will be updated after final model evaluation runs.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset by [Mart Jürisoo](https://github.com/martj42) on Kaggle and GitHub
- Apache Spark & Delta Lake open-source communities
- XGBoost and scikit-learn contributors
