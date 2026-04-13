import requests
import pandas as pd
from io import StringIO
from .CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col, isnan, to_date, when, count
from .app_constants.constants import result_map, pre_process_schema
from delta.tables import DeltaTable
from .helpers.GetEnv import GetEnv


def feature_extraction(sparkSession, dataframe):
    dataframe.createOrReplaceTempView('PreprocessTable')
    featured_result = sparkSession.sql("""

                        select count(*) cnt, home_team from preprocessTable group by home_team
                        """)
    featured_result.show()




if __name__ == '__main__':

    start_date = '1872-11-30'
    end_date = '2024-12-31'
    _env = GetEnv.get_env_variables()

    url = "https://raw.githubusercontent.com/ManikandanRJSM/international_results/master/results.csv"
    

    spark_session = SparkSessionFactory.create_spark_session()
    delta_path = f"{_env['DATA_LAKE_PATH']}/pre_processed_data/result"

    response = requests.get(url)
    pdf = pd.read_csv(StringIO(response.text))

    # Convert Pandas → Spark DataFrame
    df = spark_session.createDataFrame(pdf)

    upcoming_matches_df = df.filter(isnan(col('home_score')) & isnan(col('away_score')))

    #withColumn('date', to_date("date")) convert date as string to date
    df = df.withColumn('formated_date', to_date("date")) \
        .filter(~isnan(col('home_score')) & ~isnan(col('away_score')))
    
    # Remove the duplicate entry
    df = df.filter( (col('tournament') != 'Friendly') & (col('home_team') != 'Tahiti') & (col('away_team') != 'New Caledonia') & (col('date') != '1974-02-17') )
    
    df = df.withColumn( 'home_score', col('home_score').cast("int") ).withColumn( 'away_score', col('away_score').cast("int") )

    if end_date is not None:
        df = df.filter( col('formated_date').between(start_date, end_date) )
    
    data_df = df.withColumn(
        'match_result', when(col('home_score') > col('away_score'), result_map['home_win']) \
        .when(col('home_score') == col('away_score'), result_map['draw']) \
        .when(col('home_score') < col('away_score'), result_map['away_win'])
    )

    condition_to_check = " OR ".join([ f"target.{i} != source.{i}" for i in pre_process_schema ])

    if DeltaTable.isDeltaTable(spark_session, delta_path):
        deltaTable = DeltaTable.forPath(spark_session, delta_path)

        deltaTable.alias("target").merge(
        data_df.alias("source"),
        "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team"
        ).whenMatchedUpdate(condition=condition_to_check, set={ col: f"source.{col}" for col in pre_process_schema }) \
        .whenNotMatchedInsertAll() \
        .execute()
    else:
        data_df.write.format("delta").mode("overwrite").save(delta_path) # it saves delta log


    feature_extraction(spark_session, data_df)
        
    spark_session.stop()