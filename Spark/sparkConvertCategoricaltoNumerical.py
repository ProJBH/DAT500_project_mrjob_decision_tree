from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_replace,
    col,
    unix_timestamp,
    date_format,
    round,
    size,
    split,
)

from pyspark.ml.feature import StringIndexer
import calendar, time


def main():
    start_time = time.time()

    # loading data from HDFS
    ##################################################
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/train_full_filledMissingValues_output/part*.csv",
        1: "hdfs://namenode:9000/dis_materials/test_full_filledMissingValues_output/part*.csv",
    }
    # creating spark session
    #################################################
    spark = SparkSession.builder.appName(
        "Convert categorical to numerical"
    ).getOrCreate()

    sparkDF = spark.read.csv(
        flieList[1],
        header=True,
        inferSchema=True,
    )

    # merging multiple opening time into one column
    #################################################
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

    # preprocessing primary tag column
    #################################################
    sparkDF = sparkDF.withColumn(
        "primary_tags",
        regexp_replace(
            "primary_tags",
            '"\\"\{\\"\\"primary_tags\\"\\":\\"\\"(.*)\\"\\"\}\\""',
            '\{"primary_tags":"(.*)"}',
        ),
    )

    # countiing number of features for vendor_tag and vendor_tag_name
    #################################################
    sparkDF = sparkDF.withColumn("vendor_tag", size(split(col("vendor_tag"), r"\-")))
    sparkDF = sparkDF.withColumn(
        "vendor_tag_name", size(split(col("vendor_tag_name"), r"\-"))
    )

    for dataTypes in sparkDF.dtypes:
        if dataTypes[1] == "string":
            indexer = StringIndexer(
                inputCol=dataTypes[0], outputCol=dataTypes[0] + "_indexed"
            )
            sparkDF = indexer.fit(sparkDF).transform(sparkDF)
            sparkDF = sparkDF.drop(col(dataTypes[0]))
            sparkDF = sparkDF.withColumnRenamed(dataTypes[0] + "_indexed", dataTypes[0])

    # saving data in HDFS
    ##################################################
    sparkDF.coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true"
    ).mode("overwrite").save("/dis_materials/test_full_categoricalToNumerical_output")

    # calculate execution time
    ##################################################
    end_time = time.time() - start_time
    final_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    print("Total execution time: ", final_time)


if __name__ == "__main__":
    main()
