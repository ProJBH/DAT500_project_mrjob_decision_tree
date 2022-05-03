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
        0: "hdfs://namenode:9000/dis_materials/dipanjan/train_full_1million_categoricalToNumerical_2_output/part*.csv",
        1: "hdfs://namenode:9000/dis_materials/dipanjan/csvfiles/train_full_categoricalToNumerical.csv",
        2: "hdfs://namenode:9000/dis_materials/dipanjan/train_full_50thousand_output_categorical_to_numerical/part*.csv",
        3: "hdfs://namenode:9000/dis_materials/dipanjan/test_full_50thousand_categoricalToNumerical_output/part*.csv",
    }
    spark = SparkSession.builder.appName("how to read csv file").getOrCreate()
    sparkDF = spark.read.csv(
        flieList[0],
        header=True,
        inferSchema=True,
    )

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
    output = assembler.transform(sparkDF)
    model_df = output.select("features", "target")
    # model_df.show(1, vertical=True, truncate=False)
    trainSet, testSet = model_df.randomSplit([0.7, 0.3], seed=42)
    # output.show(2, vertical=True)
    dtc = DecisionTreeClassifier(
        labelCol="target",
        featuresCol="features",
        maxBins=99999999,
        maxDepth=4,
        impurity="entropy",
    )
    dtcModel = dtc.fit(trainSet)
    dtcPred = dtcModel.transform(testSet)
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
