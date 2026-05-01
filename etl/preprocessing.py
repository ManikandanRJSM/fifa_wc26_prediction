import requests
import pandas as pd
from io import StringIO
from CustomFactories.SparkSessionFactory import SparkSessionFactory
from pyspark.sql.functions import col, isnan, to_date, when, count, monotonically_increasing_id, lit, sum
from pyspark.sql import functions as F
from .app_constants.constants import result_map, K_map
from delta.tables import DeltaTable
from helpers.GetEnv import GetEnv
from GlobalConstants.constants import pre_process_schema
import json
import argparse


def feature_extraction(sparkSession, dataframe, data_lake_path):

    featured_delta_path = f"{data_lake_path}/pre_processed_data/featured_result"
    featured_csv_path = f"{data_lake_path}/pre_processed_data/training_dataset"

    
    get_total_wins = dataframe.filter(col('match_result').isin([1, 2])).count()
    get_total_goals = dataframe.agg(sum('total_goals')).alias('sum_goals').collect()
    get_total_goals_home = dataframe.agg(sum('home_score')).alias('sum_home_goals').collect()
    get_total_goals_away = dataframe.agg(sum('away_score')).alias('sum_away_goals').collect()
    total_matches = dataframe.count()

    # Calculate the default values if the teams dont have past 5 meetings
    global_average = round(get_total_wins/total_matches, 2)
    global_goals_socred_avg = round(get_total_goals[0][0]/total_matches, 2)


    dataframe.createOrReplaceTempView('PreprocessTable')

    team_form = sparkSession.sql(f"""
                        CREATE OR REPLACE TEMP VIEW teamHistory AS
                        -- home team row
                        SELECT
                            formated_date,
                            home_team AS team,
                            if(match_result = {result_map['home_win']}, 1, 0) AS win,
                            home_score goals,
                            away_score goal_conced
                        FROM preprocessTable

                        UNION ALL

                        -- away team row
                        SELECT
                            formated_date,
                            away_team AS team,
                            if(match_result = {result_map['away_win']}, 1, 0) AS win,
                            away_score goals,
                            home_score goal_conced
                        FROM preprocessTable     
                    """)
    

    sparkSession.sql(f"""
                        CREATE OR REPLACE TEMP VIEW teamForm AS
                        SELECT formated_date, 
                        team, 
                        coalesce(round(avg(win) over(partition by team order by formated_date rows between 6 preceding and 1 preceding), 2), {global_average}) AS win_rate_5,
                        coalesce(round(avg(goals) over(partition by team order by formated_date rows between 6 preceding and 1 preceding), 2), {global_goals_socred_avg}) AS avg_goalsrate_5,
                        coalesce(round(avg(goal_conced) over(partition by team order by formated_date rows between 6 preceding and 1 preceding), 2), ( SELECT ROUND(AVG(goal_conced), 2) AS default_conced FROM teamHistory )) AS avg_goals_conced_last_5
                        FROM teamHistory
                     
    """)
    
    featured_result = sparkSession.sql(f"""
        SELECT
            pt.*,
            home_tf.win_rate_5 AS home_team_win_rate_5,
            away_tf.win_rate_5 AS away_team_win_rate_5,
                                       
            home_tf.avg_goalsrate_5 AS home_team_avg_goals_rate_5,
            away_tf.avg_goalsrate_5 AS away_team_avg_goals_rate_5,
                                       
            home_tf.avg_goals_conced_last_5 AS home_avg_goals_conceded_last5,
            away_tf.avg_goals_conced_last_5 AS away_avg_goals_conceded_last5,
                                       
            coalesce(round(AVG(
                if(
                    (pt.home_team = Q.home_team AND Q.match_result = {result_map['home_win']}) OR
                    (pt.home_team = Q.away_team AND Q.match_result = {result_map['away_win']}),
                    1, 0
                )
            ), 2), 0.00) AS h2h_win_ratio_home,
            coalesce(round(AVG(
                Q.home_score + Q.away_score
            ), 2), 0.00) AS h2h_avg_goals

        FROM preprocessTable pt

        LEFT JOIN teamForm home_tf
            ON pt.formated_date = home_tf.formated_date
            AND pt.home_team = home_tf.team

        LEFT JOIN teamForm away_tf
            ON pt.formated_date = away_tf.formated_date
            AND pt.away_team = away_tf.team

        LEFT JOIN preprocessTable Q
            ON (
                (Q.home_team = pt.home_team AND Q.away_team = pt.away_team) OR
                (Q.home_team = pt.away_team  AND Q.away_team = pt.home_team)
            )
            AND Q.formated_date < pt.formated_date

        GROUP BY
            pt.date,
            pt.formated_date,
            pt.total_goals,
            pt.is_neutral,
            pt.match_importance,
            pt.home_team,
            pt.away_team,
            pt.home_score,
            pt.away_score,
            pt.tournament,
            pt.city,
            pt.country,
            pt.neutral,
            pt.match_result,
            home_tf.win_rate_5,
            away_tf.win_rate_5,
            home_tf.avg_goalsrate_5,
            away_tf.avg_goalsrate_5,
            home_tf.avg_goals_conced_last_5,
            away_tf.avg_goals_conced_last_5

        ORDER BY pt.formated_date
    """)

    
    # featured_result.show(2)
    featured_result = featured_result.withColumn('inc_id', monotonically_increasing_id() + 1).withColumn('home_elo', lit(None).cast("double")).withColumn('away_elo', lit(None).cast("double"))
    
    pdf = featured_result.toPandas()

    elo_dict = {}
    default_elo = 1500

    for indx, row in pdf.iterrows():

        home_team  = row['home_team']
        away_team  = row['away_team']
        tournament = row['tournament']
        result     = row['match_result']

        # step 1 — fetch elo (default 1500 if first time)
        home_elo = elo_dict.get(home_team, default_elo)
        away_elo = elo_dict.get(away_team, default_elo)

        # step 2 — calculate expected prob
        E_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

        # step 3 — store as features BEFORE updating
        pdf.at[indx, 'home_elo']      = home_elo
        pdf.at[indx, 'away_elo']      = away_elo
        pdf.at[indx, 'elo_diff']      = round(home_elo - away_elo, 2)
        pdf.at[indx, 'expected_prob'] = round(E_home, 2)

        # step 4 — actual result
        if result == result_map['home_win']:
            S_home = 1
        elif result == result_map['draw']:
            S_home = 0.5
        else:
            S_home = 0

        # step 5 — K factor
        K = K_map.get(tournament, 20)

        # step 6 — update elo AFTER match
        elo_dict[home_team] = round(float(home_elo + K * (S_home - E_home)), 2)
        elo_dict[away_team] = round(float(away_elo + K * ((1 - S_home) - (1 - E_home))), 2)

    feature_df = sparkSession.createDataFrame(pdf)

    feature_df = feature_df.withColumn( 'home_score', col('home_score').cast("int") ).withColumn( 'away_score', col('away_score').cast("int") )
  
    feature_df.write.format('csv').mode("overwrite").option("header", True).save(featured_csv_path)
    feature_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(featured_delta_path) # option("mergeSchema", "true") for schema evolution

    # if DeltaTable.isDeltaTable(spark_session, delta_path):
    #     deltaTable = DeltaTable.forPath(spark_session, delta_path)

    #     deltaTable.alias("target").merge(
    #     data_df.alias("source"),
    #     "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team AND target.match_result = source.match_result"
    #     ).whenMatchedUpdate(condition=condition_to_check, set={ col: f"source.{col}" for col in pre_process_schema }) \
    #     .whenNotMatchedInsertAll() \
    #     .execute()
    # else:
    #     data_df.write.format("delta").mode("overwrite").save(delta_path) # it saves delta log
        
    with open(f'{data_lake_path}/pre_processed_data/elo/elo.json', 'w') as f:
        json.dump(elo_dict, f, indent=2)

    # featured_result.printSchema()
    print("Preprocessing Done............!")

        


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", required=True, type=str, choices=['test', 'train'], default='train', help="Preprocess data for train / test")
    # parser.add_argument("--start_date", required=True, type=str, default='1872-11-30', help="Start date")
    # parser.add_argument("--end_date", required=True, type=str, default='2024-12-31', help="End date")

    # args = parser.parse_args()

    # mode = args.mode
    # start_date = args.start_date
    # end_date = args.end_date
    _env = GetEnv.get_env_variables()

    url = "https://raw.githubusercontent.com/ManikandanRJSM/international_results/master/results.csv"
    

    spark_session = SparkSessionFactory.create_spark_session()

    # if mode == 'test':
    #     delta_path = f"{_env['DATA_LAKE_PATH']}/pre_processed_data/test_data"
    # else:
    delta_path = f"{_env['DATA_LAKE_PATH']}/pre_processed_data/preprocessed_result"

    response = requests.get(url)
    pdf = pd.read_csv(StringIO(response.text))
    mapping_expr = F.create_map([lit(x) for kv in K_map.items() for x in kv])

    # Convert Pandas -> Spark DataFrame
    df = spark_session.createDataFrame(pdf)

    upcoming_matches_df = df.filter(isnan(col('home_score')) & isnan(col('away_score')))

    #withColumn('date', to_date("date")) convert date as string to date
    df = df.withColumn('formated_date', to_date("date")) \
        .filter(~isnan(col('home_score')) & ~isnan(col('away_score')))
    
    # Remove the duplicate entry
    cleaned_df = df.dropDuplicates()
    
    cleaned_df = cleaned_df.withColumns( {
        'home_score' : col('home_score').cast("int"),
        'away_score' : col('away_score').cast("int")
    })
    
    # Quarantine DF
    quarantine_df = df.exceptAll(cleaned_df)

    # if end_date is not None:
    #     cleaned_df = cleaned_df.filter( col('formated_date').between(start_date, end_date) )
    
    cleaned_df = cleaned_df.withColumn(
        'match_result', when(col('home_score') > col('away_score'), result_map['home_win']) \
        .when(col('home_score') == col('away_score'), result_map['draw']) \
        .when(col('home_score') < col('away_score'), result_map['away_win'])
    )

    data_df = cleaned_df.withColumns({
        'total_goals' : col('home_score') + col('away_score'),
        'is_neutral' : when(col('neutral') == True, 1).otherwise(0),
        'match_importance' : F.coalesce(mapping_expr[col('tournament')], lit(20))
    })

    condition_to_check = " OR ".join([ f"target.{i} != source.{i}" for i in pre_process_schema ])

    if DeltaTable.isDeltaTable(spark_session, delta_path):
        deltaTable = DeltaTable.forPath(spark_session, delta_path)

        deltaTable.alias("target").merge(
        data_df.alias("source"),
        "target.date = source.date AND target.home_team = source.home_team AND target.away_team = source.away_team AND target.match_result = source.match_result"
        ).whenMatchedUpdate(condition=condition_to_check, set={ col: f"source.{col}" for col in pre_process_schema }) \
        .whenNotMatchedInsertAll() \
        .execute()
    else:
        data_df.write.format("delta").mode("overwrite").save(delta_path) # it saves delta log


    feature_extraction(spark_session, data_df, _env['DATA_LAKE_PATH'])
        
    spark_session.stop()
