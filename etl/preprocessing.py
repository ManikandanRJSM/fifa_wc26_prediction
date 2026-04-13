import requests
import pandas as pd
from io import StringIO
from .CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col, isnan, to_date, when
from .app_constants.constants import result_map
from delta.tables import DeltaTable
from .helpers.GetEnv import GetEnv





if __name__ == '__main__':

    start_date = '1872-11-30'
    end_date = '2024-12-31'
    _env = GetEnv.get_env_variables()

    url = "https://raw.githubusercontent.com/ManikandanRJSM/international_results/master/results.csv"
    

    spark_session = SparkSessionFactory.create_spark_session()
    deltaTable = DeltaTable.forPath(spark_session, f"{_env['DATA_LAKE_PATH']}/processed_data")

    response = requests.get(url)
    pdf = pd.read_csv(StringIO(response.text))

    # Convert Pandas → Spark DataFrame
    df = spark_session.createDataFrame(pdf)

    upcoming_matches_df = df.filter(isnan(col('home_score')) & isnan(col('away_score')))

    #withColumn('date', to_date("date")) convert date as string to date
    df = df.withColumn('formated_date', to_date("date")) \
        .filter(~isnan(col('home_score')) & ~isnan(col('away_score')))

    if end_date is not None:
        df = df.withColumn( 'home_score', col('home_score').cast("int") ) \
        .withColumn( 'away_score', col('away_score').cast("int") ) \
        .filter( col('formated_date').between(start_date, end_date) )
    
    data_df = df.withColumn(
        'match_result', when(col('home_score') > col('away_score'), result_map['home_win']) \
        .when(col('home_score') == col('away_score'), result_map['draw']) \
        .when(col('home_score') < col('away_score'), result_map['away_win'])
    )
    print(data_df.printSchema())

    deltaTable.alias("target").merge(
    data_df.alias("source"),
    "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team"
    ).whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()
        
    spark_session.stop()