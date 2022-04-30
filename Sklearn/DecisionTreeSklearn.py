import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# https://medium.com/themlblog/splitting-csv-into-train-and-test-data-1407a063dd74
# https://realpython.com/train-test-split-python-data/
# https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
# https://www.askpython.com/python/examples/decision-trees
# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# https://www.geeksforgeeks.org/how-to-convert-categorical-string-data-into-numeric-in-python/


def miscellaneous(df):
    # save all datatypes into a variable
    datatype = df.dtypes
    # print all columns with specific datatypes, converts into list
    var1 = df.select_dtypes(include=["object"]).columns
    # print all columns with specific datatypes, converts into list
    var2 = datatype[(datatype == "object") | (datatype == "category")].index.tolist()
    # print column datatypes
    var3 = df.dtypes["sunday_from_time2"]
    # print specific row with specific column
    var4 = print(df.loc[88, "language"])
    # print column data
    var5 = df[["sunday_from_time2", "status_x"]]
    # print number of null data in total dataframe
    var6 = df.isnull().sum().sum()


def preProcessing(dataFrame, columnName, columnType, replaceStrategy):

    # if column type is object then fill all the missing values with most frequent method
    if columnType == "object":
        imputer = SimpleImputer(
            missing_values=np.nan, strategy=replaceStrategy, fill_value=None
        )
    # if column type is int or float then fill all the missing values with mean method
    elif columnType == "int64" or "float":
        imputer = SimpleImputer(
            missing_values=np.nan, strategy=replaceStrategy, fill_value=None
        )

    # transforming missing values into the dataframe
    imputer = imputer.fit(dataFrame[[columnName]])
    dataFrame[columnName] = imputer.transform(dataFrame[[columnName]])
    # converting categorical data into numerical
    le = LabelEncoder()
    if columnType == "object":
        label = le.fit_transform(dataFrame[columnName])
        dataFrame.drop(columnName, axis=1, inplace=True)
        dataFrame[columnName] = label

    return dataFrame


def main():
    fileName = r"E:\DipanjanDocuments\Education\University of Stavanger\Spring - 2022\DAT500 - Data Intensive System\Project\\train_full_50thousand.csv"
    fileName1 = r"E:\DipanjanDocuments\Education\University of Stavanger\Spring - 2022\DAT500 - Data Intensive System\Project\\test_full_50thousand.csv"

    df = pd.read_csv(fileName)
    print("training set: ", len(df), " rows")

    # print("before conversion:\n", df[["sunday_from_time2", "status_x"]].head(5))

    # pre processing the dataset
    showColumnDataTypesGroupByDtypes = df.columns.to_series().groupby(df.dtypes).groups
    for keys, values in showColumnDataTypesGroupByDtypes.items():
        if keys == "object":
            for columnName in values:
                df = preProcessing(df, columnName, keys, "most_frequent")
        elif keys == "int64" or keys == "float64":
            for columnName in values:
                df = preProcessing(df, columnName, keys, "median")

    # print("after conversion:\n", df[["sunday_from_time2", "status_x"]].head(5))
    # splitting dataset
    X = df.drop("target", axis=1)
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    fileName1 = r"E:\DipanjanDocuments\Education\University of Stavanger\Spring - 2022\DAT500 - Data Intensive System\Project\\test_full_50thousand.csv"
    df1 = pd.read_csv(fileName1)
    showColumnDataTypesGroupByDtypes1 = (
        df1.columns.to_series().groupby(df1.dtypes).groups
    )
    for keys, values in showColumnDataTypesGroupByDtypes1.items():
        if keys == "object":
            for columnName in values:
                df1 = preProcessing(df1, columnName, keys, "most_frequent")
        elif keys == "int64" or keys == "float64":
            for columnName in values:
                df1 = preProcessing(df1, columnName, keys, "median")

    # make decision tree using criterion = gini
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_pred = clf.predict(df1)
    df1["target"] = test_pred
    print(df1)
    df1.to_csv("E:\\Downloads\\abcd.csv")

    # print("y_pred: ", y_pred)
    # print("y_test: ", y_test)
    # print("Accuracy gini:", metrics.accuracy_score(y_test, y_pred))

    # # make decision tree using criterion = entropy and max depth = 3
    # clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    # clf2 = clf2.fit(X_train, y_train)
    # y_pred2 = clf2.predict(X_test)
    # print("Accuracy entropy:", metrics.accuracy_score(y_test, y_pred2))

    # accuracy is better in entropy


if __name__ == "__main__":
    main()
