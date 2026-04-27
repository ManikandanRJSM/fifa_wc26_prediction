import pandas as pd
from delta.tables import DeltaTable
from helpers.GetEnv import GetEnv
from GlobalConstants.constants import x_training_schema, y_training_schema
from CustomFactories.SparkSessionFactory import SparkSessionFactory

import xgboost as xgb

import joblib


if __name__ == "__main__":
    _env = GetEnv.get_env_variables()
    spark_session = SparkSessionFactory.create_spark_session()

    df = spark_session.read.format('delta').load(f"{_env['DATA_LAKE_PATH']}/pre_processed_data/featured_result")

    pd = df.toPandas()
    
    X_train = pd[x_training_schema]
    y_train = pd[y_training_schema]


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
        verbose  = 50                   # print every 50 trees
    )

    joblib.dump(model, "fifa_xgb_model.pkl")

    print("Model saved!")