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
    lit,
    rand,
    when,
    monotonically_increasing_id,
    row_number,
)
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import pandas as pd
import time


def main():
    start_time = time.time()
    flieList = {
        0: "hdfs://namenode:9000/dis_materials/dipanjan/csvfiles/test_full_categoricalToNumerical.csv",
        1: "hdfs://namenode:9000/dis_materials/dipanjan/csvfiles/train_full_categoricalToNumerical.csv",
        2: "hdfs://namenode:9000/dis_materials/dipanjan/train_full_50thousand_output_categorical_to_numerical/part*.csv",
        3: "hdfs://namenode:9000/dis_materials/dipanjan/test_full_50thousand_categoricalToNumerical_output/part*.csv",
    }
    spark = SparkSession.builder.appName("how to read csv file").getOrCreate()
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

    # nums = np.random.choice([0, 1], size=1000, p=[0.2, 0.8])
    # nums1 = np.random.binomial(n=1, p=0.9, size=[1000])
    # df = pd.DataFrame(np.random.binomial(n=1, p=0.9, size=p), columns=(["target"]))
    df = pd.DataFrame(
        np.random.choice([0, 1], size=p, p=[0.9, 0.1]), columns=(["target"])
    )
    df["target"] = df["target"].astype(np.double)
    sparkDFTemp = spark.createDataFrame(df)
    # sparkDFTemp.printSchema()
    sparkDFTest = sparkDFTest.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    sparkDFTemp = sparkDFTemp.withColumn(
        "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
    )
    sparkDFTest = sparkDFTest.join(sparkDFTemp, on=["row_index"]).drop("row_index")
    # sparkDFTest.show(1, vertical=True, truncate=False)
    # sparkDFTest = sparkDFTest.withColumn(
    #     "target", when(rand() > 0.5, 1).otherwise(0).cast(DoubleType())
    # )
    sparkDFTest.groupBy("target").count().show()
    # modelDFTrain = sparkDFTest.select("target")
    # modelDFTrain.show()
    # abc = spark.createDataFrame(df)

    # sparkDFTest = sparkDFTest.withColumn("target", lit(1))
    # sparkDFTest = sparkDFTest.withColumn("target", col("target").cast(DoubleType()))

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

    assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")
    outputTrain = assembler.transform(sparkDFTrain)

    modelDFTrain = outputTrain.select("features", "target")

    dtc = DecisionTreeClassifier(labelCol="target", featuresCol="features")
    dtcModel = dtc.fit(modelDFTrain)
    outputTest = assembler.transform(sparkDFTest)
    modelDFTest = outputTest.select("features", "target")
    dtcPred = dtcModel.transform(modelDFTest)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy"
    )
    dtcAcc = evaluator.evaluate(dtcPred)
    print("accuracy", dtcAcc)
    end_time = time.time() - start_time
    final_time = time.strftime("%H:%M:%S", time.gmtime(end_time))
    print("Total execution time: ", final_time)


if __name__ == "__main__":
    main()
