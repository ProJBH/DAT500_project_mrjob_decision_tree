from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_replace,
    col,
    isnan,
    when,
    count,
    to_timestamp,
    from_unixtime,
    unix_timestamp,
    date_format,
    round,
    size,
    split,
)
from pyspark.sql.types import StringType, DoubleType, IntegerType, TimestampType
from sklearn.impute import SimpleImputer
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    OneHotEncoder,
    VectorIndexer,
)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector
import numpy as np
from pyspark.sql.types import DateType
import calendar, time


def main():
    start_time = time.time()
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/dipanjan/train_full_1million_filledMissingValues_5_output/part*.csv",
        1: "hdfs://namenode:9000/dis_materials/dipanjan/test_full_50thousand_filledMissingValues_output/part*.csv",
        2: "hdfs://namenode:9000/dis_materials/dipanjan/test_full_836000rows_filledMissingValues_1_output/part*.csv",
    }
    spark = SparkSession.builder.appName("how to read csv file").getOrCreate()
    sparkDF = spark.read.csv(
        flieList[1],
        header=True,
        inferSchema=True,
    )

    # sparkDF.show(1, vertical=True)
    # sparkDF.printSchema()

    # temFri = ((friday_to_time1-friday_from_time1)+(friday_to_time2-friday_from_time2))
    #             temFri = temFri.total_seconds()/3600

    for eachDay in calendar.day_name:
        eachDay = eachDay.lower()
        subset = [
            eachDay + "_from_time2",
            eachDay + "_from_time1",
            eachDay + "_to_time2",
            eachDay + "_to_time1",
        ]
        for col_name in subset:
            sparkDF = sparkDF.withColumn(
                col_name, date_format(col(col_name), "HH:mm:ss").cast("timestamp")
            )
        sparkDF = sparkDF.withColumn(
            eachDay + "_opening_time",
            round(
                (
                    unix_timestamp(col(subset[3]))
                    - unix_timestamp(col(subset[1]))
                    + unix_timestamp(col(subset[2]))
                    - unix_timestamp(col(subset[0]))
                )
                / 3600,
                2,
            ),
        )
        for col_name in subset:
            sparkDF = sparkDF.drop(col_name)

    # sparkDF.select(
    #     "monday_opening_time",
    #     "tuesday_opening_time",
    #     "wednesday_opening_time",
    #     "thursday_opening_time",
    #     "friday_opening_time",
    #     "saturday_opening_time",
    #     "sunday_opening_time",
    # ).show(5)

    sparkDF = sparkDF.withColumn(
        "primary_tags",
        regexp_replace(
            "primary_tags",
            '"\\"\{\\"\\"primary_tags\\"\\":\\"\\"(.*)\\"\\"\}\\""',
            '\{"primary_tags":"(.*)"}',
        ),
    )
    sparkDF = sparkDF.withColumn("vendor_tag", size(split(col("vendor_tag"), r"\-")))
    sparkDF = sparkDF.withColumn(
        "vendor_tag_name", size(split(col("vendor_tag_name"), r"\-"))
    )
    # sparkDF.select(
    #     "vendor_tag_name",
    #     "vendor_tag_name_sum",
    # ).show(1, truncate=False)

    for dataTypes in sparkDF.dtypes:
        if dataTypes[1] == "string":
            indexer = StringIndexer(
                inputCol=dataTypes[0], outputCol=dataTypes[0] + "_indexed"
            )
            sparkDF = indexer.fit(sparkDF).transform(sparkDF)
            sparkDF = sparkDF.drop(col(dataTypes[0]))
            sparkDF = sparkDF.withColumnRenamed(dataTypes[0] + "_indexed", dataTypes[0])

    # sparkDF.show(2, vertical=True, truncate=False)

    # sparkDF.groupBy("target").count().show()
    # print(sparkDF.columns)

    # tempu = []

    # columnlen = len(sparkDF.dtypes)
    # print(columnlen)
    # sparkDF.printSchema()
    sparkDF.coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true"
    ).mode("overwrite").save(
        "/dis_materials/dipanjan/test_full_836000rows_categoricalToNumerical_1_output"
    )

    end_time = time.time() - start_time
    final_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    print("Total execution time: ", final_time)


if __name__ == "__main__":
    main()
