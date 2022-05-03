from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, isnan, when, count
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.ml.feature import Imputer
import time
import calendar

# function for counting empty cells in a column
def countIfColumnisNull(sparkDF, coLumnName):
    checkNumberofEmptyCells = sparkDF.select(
        [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in sparkDF.columns]
    )
    totalEmptyCellCount = checkNumberofEmptyCells.collect()[0][coLumnName]
    # print("Total emtpty cell in", coLumnName, ": ", z)
    return totalEmptyCellCount


def sparkPreProcessing(sparkDF):
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

    # replace vendor_tag from , to -
    ##################################################
    sparkDF = sparkDF.withColumn("vendor_tag", regexp_replace("vendor_tag", ",", "-"))
    # sparkDF.select("vendor_tag").show(5)

    # replace vendor_tag_name from , to -
    ##################################################
    sparkDF = sparkDF.withColumn(
        "vendor_tag_name", regexp_replace("vendor_tag_name", ",", "-")
    )

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

    # replace string column value with most frequentValue
    # replace number column value with median value
    ##################################################
    for dataTypes in sparkDF.dtypes:
        if dataTypes[1] == "string":
            frequentValueinColumnName = sparkDF.freqItems([dataTypes[0]], 0.60)
            frequentValue = frequentValueinColumnName.collect()[0][
                dataTypes[0] + "_freqItems"
            ]
            if len(frequentValue) == 0:
                continue
            else:

                sparkDF = sparkDF.fillna(str(frequentValue[0]), subset=[dataTypes[0]])
        elif dataTypes[1] == "int":

            imputer = Imputer(
                inputCols=[dataTypes[0]],
                outputCols=[dataTypes[0]],
            ).setStrategy("median")
            sparkDF = imputer.fit(sparkDF).transform(sparkDF)
        elif dataTypes[1] == "double":

            imputer = Imputer(
                inputCols=[dataTypes[0]],
                outputCols=[dataTypes[0]],
            ).setStrategy("median")
            sparkDF = imputer.fit(sparkDF).transform(sparkDF)

    # saving data in HDFS
    ##################################################
    sparkDF.coalesce(1).write.format("com.databricks.spark.csv").option(
        "header", "true"
    ).mode("overwrite").save(
        "/dis_materials/dipanjan/test_full_836000rows_filledMissingValues_1_output"
    )


def main():
    startTime = time.time()

    # loading data from HDFS
    ##################################################
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/train_full.csv",
        1: "hdfs://namenode:9000/dis_materials/test_full.csv",
    }

    # creating spark session
    ##################################################
    spark = SparkSession.builder.appName("Fill missing value").getOrCreate()
    sparkDF = spark.read.csv(
        flieList[0],
        header=True,
        inferSchema=True,
    )

    # initialize preprocessing function
    ##################################################
    sparkPreProcessing(sparkDF)

    # calculate execution time
    ##################################################
    endTime = time.time() - startTime
    finalTime = time.strftime("%H:%M:%S", time.gmtime(endTime))
    print("Total execution time: ", finalTime)


if __name__ == "__main__":
    main()
