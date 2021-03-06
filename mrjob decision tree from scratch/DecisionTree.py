#!/usr/bin/env python3
import numpy as np
import random

# Split training dataset, let's parts of data as test set
def train_test_split(df, sub_size):
    if isinstance(sub_size, float):
        test_size = round(sub_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    train_df = df.loc[test_indices]
    test_df = df.drop(test_indices)
    return train_df, test_df


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    return classification


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    # excluding the last column which is the label
    for column_index in range(n_columns - 1):
        values = data[:, column_index]
        unique_values = np.unique(values)
        potential_splits[column_index] = unique_values
    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
    # feature is categorical
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    return data_below, data_above


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))
    return overall_entropy


def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(
                data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(
                data_below, data_above)
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "target":
            unique_values = df[feature].unique()
            example_value = unique_values[0]
            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    return feature_types

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification
    # recursive part
    else:
        counter += 1
        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(
            data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        # instantiate sub-tree
        sub_tree = {question: []}
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(
            data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(
            data_above, counter, min_samples, max_depth)
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree


def make_prediction(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")
    # ask question
    try:
        if comparison_operator == "<=":  # feature is continuous
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        # feature is categorical
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        # base case
        if not isinstance(answer, dict):
            return answer
        # recursive part
        else:
            residual_tree = answer
            return make_prediction(example, residual_tree)
    except TypeError:
        print("Try to compare: Column name: ", feature_name, " Value is: ", example[feature_name], " Type is: ", type(example[feature_name])," Operator is ", comparison_operator, " with ", value, "as float")
        return      

def calculate_accuracy(df, tree):
    df["classification"] = df.apply(make_prediction, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["target"]
    accuracy = df["classification_correct"].mean()
    return accuracy

