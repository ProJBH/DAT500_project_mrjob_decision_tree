import string
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, isnan, when, count, isnull
from pyspark.sql.types import StringType, DoubleType, IntegerType

from pyspark.ml.feature import Imputer
import time
import calendar


def pre_processing(df):
    # df.drop
    # drop any id related columns

    # counted with how many different numbers inside of the region
    # vendor_tag_name should be removed

    weekdays = [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ]
    for weekday in weekdays:
        df[weekday + "_from_time2"] = df[weekday + "_from_time2"].fillna("00:00:00")
        df[weekday + "_from_time1"] = df[weekday + "_from_time1"].fillna("00:00:00")
        df[weekday + "_to_time2"] = df[weekday + "_to_time2"].fillna("00:00:00")
        df[weekday + "_to_time1"] = df[weekday + "_to_time1"].fillna("00:00:00")

    df["vendor_tag"] = df["vendor_tag"].str.replace(",", "-")
    df["vendor_tag_name"] = df["vendor_tag_name"].str.replace(",", "-")

    # print(df["vendor_tag_name"])
    # print(df["vendor_tag"])

    # drop if 70% of value are null

    # fill in missing value
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in list(df.select_dtypes(include=numerics)):
        df[col].fillna(value=df[col].median(), inplace=True)
    for col in list(df.select_dtypes(["object"])):
        df[col].fillna(value=df[col].mode()[0], inplace=True)
    df.to_csv("/home/ubuntu/dipanjan/train_full_preProcessed_0_0_0.csv")


def countIfColumnisNull(sparkDF, coLumnName):
    checkNumberofEmptyCells = sparkDF.select(
        [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sparkDF.columns]
    )
    totalEmptyCellCount = checkNumberofEmptyCells.collect()[0][coLumnName]
    # print("Total emtpty cell in", coLumnName, ": ", z)
    return totalEmptyCellCount


def sparkPreProcessing(sparkDF):
    # sparkDF["vendor_tag"] = sparkDF["vendor_tag"].str.replace(",", "-")
    # print(sparkDF["vendor_tag"])
    # sparkDF.select("vendor_tag").str.replace(",", "-")

    # saving column data type information in dictonary
    ##################################################
    stringColumns = [
        f.name for f in sparkDF.schema.fields if isinstance(f.dataType, StringType)
    ]
    doubleColumns = [
        f.name for f in sparkDF.schema.fields if isinstance(f.dataType, DoubleType)
    ]
    integerColumns = [
        f.name for f in sparkDF.schema.fields if isinstance(f.dataType, IntegerType)
    ]

    columnDataTypes = {
        "string": stringColumns,
        "double": doubleColumns,
        "int": integerColumns,
    }

    # for columnType, columnName in columnDataTypes.items():
    #     for x in columnName:
    #         if columnType == "string":
    #             imputer = Imputer(
    #                 inputCols=[x],
    #                 # outputCols=["{}_imputed".format(c) for c in ["status_x"]]
    #                 outputCols=[x],
    #             ).setStrategy("mode")
    #             sparkDF = imputer.fit(sparkDF).transform(sparkDF)

    # countIfColumnisNull(sparkDF, "customer_id")

    # replace vendor_tag from , to -
    ##################################################
    sparkDF = sparkDF.withColumn("vendor_tag", regexp_replace("vendor_tag", ",", "-"))
    # sparkDF.select("vendor_tag").show(5)

    # replace vendor_tag_name from , to -
    ##################################################
    sparkDF = sparkDF.withColumn(
        "vendor_tag_name", regexp_replace("vendor_tag_name", ",", "-")
    )
    # sparkDF.select("vendor_tag_name").show(10)

    # replace empty date time column value with 00:00:00
    ##################################################
    for eachDay in calendar.day_name:
        eachDay = eachDay.lower()
        sparkDF = sparkDF.fillna(
            value="00:00:00",
            subset=[
                eachDay + "_from_time2",
                eachDay + "_from_time1",
                eachDay + "_to_time2",
                eachDay + "_to_time1",
            ],
        )
    # sparkDF.select("sunday_from_time2").show(10)
    # sparkDF.repartition(1).write.format("com.databricks.spark.csv").save(
    #     "/dis_materials/dipanjan/test", header="true"
    # )

    # sparkDF.write.format("com.databricks.spark.csv").save(
    #     "/home/ubuntu/dipanjan/testfile.csv"
    # )

    for dataTypes in sparkDF.dtypes:
        if dataTypes[1] == "string":
            # if countIfColumnisNull(sparkDF, dataTypes[0]) == 0:
            # continue
            # else:
            frequentValueinColumnName = sparkDF.freqItems([dataTypes[0]], 0.60)
            # frequentValueinColumn.show(truncate=False)
            frequentValue = frequentValueinColumnName.collect()[0][
                dataTypes[0] + "_freqItems"
            ]
            if len(frequentValue) == 0:
                # print(frequentValueinColumnName)
                continue
            else:
                # print("Most frequent value found for column ", dataTypes[0], "is: ", k)
                sparkDF = sparkDF.fillna(str(frequentValue[0]), subset=[dataTypes[0]])
        elif dataTypes[1] == "int":
            # print(dataTypes[0])
            imputer = Imputer(
                inputCols=[dataTypes[0]],
                # outputCols=["{}_imputed".format(c) for c in ["status_x"]]
                outputCols=[dataTypes[0]],
            ).setStrategy("median")
            sparkDF = imputer.fit(sparkDF).transform(sparkDF)
        elif dataTypes[1] == "double":
            # print(dataTypes[0])
            imputer = Imputer(
                inputCols=[dataTypes[0]],
                # outputCols=["{}_imputed".format(c) for c in ["status_x"]]
                outputCols=[dataTypes[0]],
            ).setStrategy("median")
            sparkDF = imputer.fit(sparkDF).transform(sparkDF)
        # countIfColumnisNull(sparkDF, dataTypes[0])
        # print("column name: ", dataTypes[0], " column type: ", dataTypes[1])

    # # Add imputation cols to df
    # sparkDF = imputer.fit(sparkDF).transform(sparkDF)

    # for dataTypes in sparkDF.dtypes:
    #     # print(dataTypes)
    #     if dataTypes[1] == "string":
    #         imputer = Imputer(
    #             inputCols=[dataTypes[0]],
    #             outputCols=[dataTypes[0] + "_imputed"],
    #         ).setStrategy("mode")
    #         df = imputer.fit(sparkDF).transform(sparkDF)
    #         print(df)

    # for columnType, columnName in columnDataTypes.items():
    #     if columnType == "integer":
    #         imputer = Imputer(
    #             inputCols=[columnName],
    #             outputCols=[columnName],
    #         ).setStrategy("mode")

    # imputer = Imputer(
    #     inputCols=["customer_id"],
    #     # outputCols=["{}_imputed".format(c) for c in ["status_x"]]
    #     outputCols=["customer_id"],
    # ).setStrategy("median")

    # # # # Add imputation cols to df
    # sparkDF = imputer.fit(sparkDF).transform(sparkDF)

    # frequentValueinColumn = sparkDF.freqItems(["created_at_x"], 0.60)
    # # frequentValueinColumn.show(truncate=False)
    # k = frequentValueinColumn.collect()[0]["created_at_x_freqItems"]
    # print(k)

    # sparkDF = sparkDF.fillna(k[0], subset=["customer_id"])
    # sparkDF.select("created_at_x").show(5)
    # counting null value in all the dataframe
    ################################################
    # for dataTypes in sparkDF.dtypes:
    #     z = countIfColumnisNull(sparkDF, dataTypes[0])
    #     print("total empty cell in ", dataTypes[0], " is: ", z)

    # sparkDF.coalesce(1).write.mode("overwrite").option("header", "true").csv(
    #     "hdfs://namenode:9000/dis_materials/dipanjan/test_file.csv"
    # )

    # sparkDF.write.option("header", "true").csv("/home/ubuntu/dipanjan/testfile.csv")

    sparkDF.coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true"
    ).mode("overwrite").save(
        "/dis_materials/dipanjan/test_full_836000rows_filledMissingValues_1_output"
    )


def main():
    startTime = time.time()
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/train_full_0_0_0.csv",
        1: "hdfs://namenode:9000/dis_materials/project_data/train_full.csv",
        2: "hdfs://namenode:9000/dis_materials/train_full_preProcessed_0_0_0.csv",
        3: "hdfs://namenode:9000/dis_materials/project_data/train_full_200thousand.csv",
        4: "hdfs://namenode:9000/dis_materials/project_data/train_full_50thousand.csv",
        5: "hdfs://namenode:9000/dis_materials/project_data/test_full_50thousand.csv",
        6: "hdfs://namenode:9000/dis_materials/project_data/train_full_1million_5.csv",
        7: "hdfs://namenode:9000/dis_materials/project_data/test_full_836000rows_1.csv",
    }
    spark = SparkSession.builder.appName("how to read csv file").getOrCreate()
    sparkDF = spark.read.csv(
        flieList[7],
        header=True,
        inferSchema=True,
    )

    # columnlen = len(sparkDF.dtypes)
    # print(columnlen)

    # print spark schema
    # sparkDF.printSchema()
    sparkPreProcessing(sparkDF)

    endTime = time.time() - startTime
    finalTime = time.strftime("%H:%M:%S", time.gmtime(endTime))
    print("Total execution time: ", finalTime)

    # pandas preprocessing
    # df = pd.read_csv("/home/ubuntu/train_full_0_0_0.csv")
    # pre_processing(df)


if __name__ == "__main__":
    main()
