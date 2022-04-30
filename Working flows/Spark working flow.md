# Working with spark

## **Install required packages**
```console
pip install pyspark
```

## **Working with dataset**
## 1. Testing dataset
### Fill missing values

- Input file name & location in HDFS: _hdfs:9000//dis_materials/project_data/test_full.csv_
- Fill the missing value with driver program, located in: _/home/ubuntu/dipanjan/sparkFillMissingValues.py_ 
- Start the Program from linux terminal: `spark-submit --master yarn --executor-cores 4 --num-executors 4 --executor-memory 6g --driver-memory 1g --conf spark.kryoserializer.buffer.max=512m sparkFillMissingValues.py`
    > We can run it locally if we want but it will not use any haddoop cluster during runtime: `python3 sparkFillMissingValues.py`
- After completing the task, the result will be saved in HDFS location: _hdfs:9000//dis_materials/project_data/test_full_preprocessed.csv_

### Convert categorical data to numerical

- We used the output result from the previous section where empty or null data is being filled up with two different technique. The input file location is hard coded in driver program.
- Now the dataset is filled up and we will convert it to numerical data because before applying the machine learning algorithm all the data should be in numerical format.
- The driver program located in: _/home/ubuntu/dipanjan/sparkConvertCategoricaltoNumerical.py_
- Start the Program from linux terminal: `spark-submit --master yarn --executor-cores 4 --num-executors 4 --executor-memory 6g --driver-memory 1g --conf spark.kryoserializer.buffer.max=512m sparkConvertCategoricaltoNumerical.py`
    > We can run it locally if we want but it will not use any haddoop cluster during runtime: `python3 sparkConvertCategoricaltoNumerical.py`
- After completing the task, the result will be saved in HDFS location: _hdfs:9000//dis_materials/project_data/test_full_preprocessed_categoricalToNumerical.csv_

## 2. Training dataset
_This section is identical to the previous section(1. Testing dataset) but we are now doing the same thing for our training dataset_
### Fill missing values

- Input file name & location in HDFS: _hdfs:9000//dis_materials/project_data/train_full.csv_
- Fill the missing value with driver program, located in: _/home/ubuntu/dipanjan/sparkFillMissingValues.py_ 
- Start the Program from linux terminal: `spark-submit --master yarn --executor-cores 4 --num-executors 4 --executor-memory 6g --driver-memory 1g --conf spark.kryoserializer.buffer.max=512m sparkFillMissingValues.py`
    > We can run it locally if we want but it will not use any haddoop cluster during runtime: `python3 sparkFillMissingValues.py`
- After completing the task, the result will be saved in HDFS location: _hdfs:9000//dis_materials/project_data/train_full_preprocessed.csv_

### Convert categorical data to numerical

- We used the output result from the previous section where empty or null data is being filled up with two different technique. The input file location is hard coded in driver program.
- Now the dataset is filled up and we will convert it to numerical data because before applying the machine learning algorithm all the data should be in numerical format.
- The driver program located in: _/home/ubuntu/dipanjan/sparkConvertCategoricaltoNumerical.py_
- Start the Program from linux terminal: `spark-submit --master yarn --executor-cores 4 --num-executors 4 --executor-memory 6g --driver-memory 1g --conf spark.kryoserializer.buffer.max=512m sparkConvertCategoricaltoNumerical.py`
    > We can run it locally if we want but it will not use any haddoop cluster during runtime: `python3 sparkConvertCategoricaltoNumerical.py`
- After completing the task, the result will be saved in HDFS location: _hdfs:9000//dis_materials/project_data/train_full_preprocessed_categoricalToNumerical.csv_


## **Apply decision tree from Spark Machine Learning Library (MLlib)**

- After getting the preprocessed data from testing and training set we applied the decision tree algorithm from mllib.classification
- We worked with both testing and training set and made a driver program to get the final outcome from the dataset
- The driver program located in: _/home/ubuntu/dipanjan/sparkDecisionTree.py_
- Start the Program from linux terminal: `spark-submit --master yarn --executor-cores 4 --num-executors 4 --executor-memory 6g --driver-memory 1g --conf spark.kryoserializer.buffer.max=512m sparkDecisionTree.py`
- Test dataset file location: _hdfs:9000//dis_materials/project_data/test_full_preprocessed_categoricalToNumerical.csv_
- Train dataset file location: _hdfs:9000//dis_materials/project_data/train_full_preprocessed_categoricalToNumerical.csv_
- Result of this program can be saved in HDFS or it is also possible to show it to the console.