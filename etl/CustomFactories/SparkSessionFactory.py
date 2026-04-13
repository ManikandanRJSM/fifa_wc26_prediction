from pyspark.sql import SparkSession

class SparkSessionFactory:
    @staticmethod
    def create_spark_session():

        #local[2] -> 2 cores if * use all cores in machine

        spark = SparkSession.builder \
                .appName("MyApp") \
                .master("local[1]") \
                .config("spark.jars", "./jars/hadoop-aws-3.4.2.jar,./jars/bundle-2.29.52.jar") \
                .config("spark.driver.extraClassPath", "./jars/hadoop-aws-3.4.2.jar:./jars/bundle-2.29.52.jar") \
                .config("spark.executor.extraClassPath", "./jars/hadoop-aws-3.4.2.jar:./jars/bundle-2.29.52.jar") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.executor.memory", "2g") \
                .getOrCreate()
        # print(spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
        # .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \ # Enable this line if you are using Delta Lake and want to use the DeltaCatalog for managing your tables. This is necessary for features like time travel, schema evolution, and other Delta Lake functionalities.
        return spark