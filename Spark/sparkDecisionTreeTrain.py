from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
    spark = SparkSession.builder.appName(
        "Decision tree - testing with training set"
    ).getOrCreate()

    sparkDF = spark.read.csv(
        flieList[1],
        header=True,
        inferSchema=True,
    )

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
    dtc = DecisionTreeClassifier(
        labelCol="target",
        featuresCol="features",
        maxBins=99999999,
        maxDepth=4,
        impurity="entropy",
    )

    # making predictions
    ##################################################
    output = assembler.transform(sparkDF)
    model_df = output.select("features", "target")
    trainSet, testSet = model_df.randomSplit([0.7, 0.3], seed=42)
    dtcModel = dtc.fit(trainSet)
    dtcPred = dtcModel.transform(testSet)

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
