# Fifa Prediction Analysis
End-to-end FIFA match outcome prediction system using ETL pipelines, feature engineering, and machine learning — powered by historical international football data and S3 as a data lake.

Data set : https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?resource=download&select=results.csv

## Setup

```bash
virtualenv venv
source venv/bin/activate
pip install pyspark==4.1.1 delta-spark==4.1.0 pandas==2.3.3 requests==2.25.1 python-dotenv==1.2.2 scikit-learn==1.7.2 xgboost==3.2.0 joblib==1.5.3
```

**Dependencies:** Python > 3.9, OpenJDK > 17, Spark==4.1.1, pyspark==4.1.1, delta-spark==4.1.0, pandas==2.3.3, requests==2.25.1, python-dotenv==1.2.2, hadoop-aws 3.4.2, scikit-learn==1.7.2, xgboost==3.2.0, joblib==1.5.3, matplotlib==3.10.9, seaborn==0.13.2, 
