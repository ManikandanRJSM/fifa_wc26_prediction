from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from helpers.GetEnv import GetEnv
import argparse
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col
from datetime import date
from GlobalConstants.constants import x_test_schema, y_test_schema
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    parser.add_argument("--start_date", required=True, type=str, default='1872-11-30', help="Start date")
    parser.add_argument("--end_date", type=str, required=False, default=str(date.today()), help="End date")

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date


    _env = GetEnv.get_env_variables()
    model_evaluation_result_path = f"{_env['DATA_LAKE_PATH']}/model_evaluation_result"

    spark_session = SparkSessionFactory.create_spark_session()
    df = spark_session.read.format('delta').load(f"{_env['DATA_LAKE_PATH']}/pre_processed_data/featured_result")

    # Get test data from start date and end with only matches with win / lose (ignores draw's since most models struggles with lose results)
    df = df.filter( col('formated_date').between(start_date, end_date) ).filter(col('match_result').isin([1, 2]))

    X_test = df.select(x_test_schema) # Only Metrics
    Y_test = df.select(y_test_schema) # Only Results
    

    X_test_pd = X_test.toPandas()
    Y_test_pd = Y_test.toPandas()

    # print(Y_test_pd.value_counts())
    # exit()


    model_path = f"{_env['DATA_LAKE_PATH']}/model/fifa_xgb_model.pkl"

    # Load saved model
    loaded_model = joblib.load(model_path)
    Y_pred = loaded_model.predict(X_test_pd)

    result_df = pd.DataFrame({
        "actual"   : Y_test_pd.squeeze(), #converts 2D dataframe (i.e rows and cols) to 1D dataframe
        "predicted": Y_pred
    })

    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(Y_test_pd, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Home Win', 'Away Win'], yticklabels=['Home Win', 'Away Win'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{model_evaluation_result_path}/confusion_matrix.png")
    plt.close()

    # Generate (Precision and Recall) and F1-Score
    report = classification_report(Y_test_pd, Y_pred, labels=[1, 2], target_names=['Home Win', 'Away Win'], output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :2].T # Transpose for better plotting
    df_report.plot(kind='bar')
    plt.title('Precision and Recall')
    plt.savefig(f"{model_evaluation_result_path}/precision_recall.png")
    plt.close()

    accuracy = accuracy_score(Y_test_pd, Y_pred)
    print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    print("Model test completed..............!")

    # Predict on test data
    # 

    # print("Actual results   :", list(Y_test_pd))
    # print("Predicted results:", list(Y_pred))