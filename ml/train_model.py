import pandas as pd
from delta.tables import DeltaTable
from helpers.GetEnv import GetEnv
from GlobalConstants.constants import x_training_schema, y_training_schema
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pathlib import Path
from pyspark.sql.functions import col
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

import joblib
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    parser.add_argument("--start_date", required=False, type=str, default='1872-11-30', help="Start date")
    parser.add_argument("--end_date", required=False, type=str, default='2024-12-31', help="End date")

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    
    _env = GetEnv.get_env_variables()

    model_path = f"{_env['DATA_LAKE_PATH']}/model/fifa_xgb_model.pkl"
    spark_session = SparkSessionFactory.create_spark_session()

    df = spark_session.read.format('delta').load(f"{_env['DATA_LAKE_PATH']}/pre_processed_data/featured_result")

    df = df.filter( col('formated_date').between(start_date, end_date) )

    pd = df.toPandas()
    
    X_train = pd[x_training_schema]
    y_train = pd[y_training_schema]

    sample_weights = compute_sample_weight('balanced', y_train)
    model = xgb.XGBClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "multi:softprob",
        num_class        = 3,
        random_state     = 42
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights
    )

    joblib.dump(model, model_path)

    print("Model saved!")