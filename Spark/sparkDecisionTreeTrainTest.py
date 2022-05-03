from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    monotonically_increasing_id,
    row_number,
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import pandas as pd
import time


def main():
    start_time = time.time()

    # loading data from HDFS
    ##################################################
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/dipanjan/csvfiles/test_full_categoricalToNumerical.csv",
        1: "hdfs://namenode:9000/dis_materials/dipanjan/csvfiles/train_full_categoricalToNumerical.csv",
    }
    # creating spark session
    ##################################################
    spark = SparkSession.builder.appName("Final decision tree").getOrCreate()

    sparkDFTrain = spark.read.csv(
        flieList[1],
        header=True,
        inferSchema=True,
    )

    sparkDFTest = spark.read.csv(
        flieList[0],
        header=True,
        inferSchema=True,
    )

    p = sparkDFTest.count()

    # making supervised dataset for testing data
    ##################################################
    df = pd.DataFrame(
        np.random.choice([0, 1], size=p, p=[0.9, 0.1]), columns=(["target"])
    )
    df["target"] = df["target"].astype(np.double)
    sparkDFTemp = spark.createDataFrame(df)
    sparkDFTest = sparkDFTest.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    sparkDFTemp = sparkDFTemp.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    sparkDFTest = sparkDFTest.join(sparkDFTemp, on=["row_index"]).drop("row_index")
    sparkDFTest.groupBy("target").count().show()

    # selecting feature columns
    ##################################################
    featureColumns = (
        "customer_id",
        "gender",
        "status_x",
        "verified_x",
        "created_at_x",
        "updated_at_x",
        "location_number",
        "location_type",
        "vendor_category_id",
        "vendor_category_en",
        "delivery_charge",
        "serving_distance",
        "sunday_opening_time",
        "monday_opening_time",
        "tuesday_opening_time",
        "wednesday_opening_time",
        "thursday_opening_time",
        "friday_opening_time",
        "saturday_opening_time",
        "is_akeed_delivering",
        "discount_percentage",
        "rank",
        "primary_tags",
        "verified_y",
        "status_y",
        "device_type",
        "latitude_x",
        "longitude_x",
        "latitude_y",
        "longitude_y",
        "created_at_y",
        "updated_at_y",
        "is_open",
        "prepration_time",
        "commission",
        "language",
        "open_close_flags",
        "vendor_tag",
        "vendor_tag_name",
        "location_number_obj",
        "vendor_rating",
    )

    # initializing vector assembler and decision tree
    ##################################################
    assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")
    dtc = DecisionTreeClassifier(labelCol="target", featuresCol="features", impurity="entropy")

    # preparing training data
    ##################################################
    outputTrain = assembler.transform(sparkDFTrain)
    modelDFTrain = outputTrain.select("features", "target")
    dtcModel = dtc.fit(modelDFTrain)

    # preparing test data
    ##################################################
    outputTest = assembler.transform(sparkDFTest)
    modelDFTest = outputTest.select("features", "target")

    # making predictions
    ##################################################
    dtcPred = dtcModel.transform(modelDFTest)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy"
    )

    # calculate accuracy
    ##################################################
    dtcAcc = evaluator.evaluate(dtcPred)
    print("accuracy", dtcAcc)

    # calculate execution time
    ##################################################
    end_time = time.time() - start_time
    final_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    print("Total execution time: ", final_time)


if __name__ == "__main__":
    main()
