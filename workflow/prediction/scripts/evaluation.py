import sys
import numpy as np
import time
import pandas as pd
from scipy import sparse

data_train_file = snakemake.input["data_train"]
labels_train_file = snakemake.input["label_train"]
data_test_file = snakemake.input["data_test"]
labels_test_file = snakemake.input["label_test"]
model = snakemake.input["model"]
method = snakemake.wildcards.method
method_config_file = snakemake.input["conf"]
group = snakemake.wildcards.group

mode = snakemake.config["MODE"]
optimization = snakemake.config["OPTIMIZATION"]

output_evaluation = snakemake.output["evaluation"]
output_fitted_model = snakemake.output["fitted_model"]


def main():
    from joblib import dump
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from skopt import BayesSearchCV
    from support import (
        preprocess,
        model_fitting,
        evaluate_classifier,
        evaluate_regression,
        save_regression_test_values,
    )
    import utility
    import pandas as pd


    #label.set_index(label.columns[0], inplace=True, drop=True)
    config_file = utility.config_reader(method_config_file)

    print("Mode = ", mode)
    print("Method = ", method)
    print("Config file = ", method_config_file)
    print("Optimization = ", optimization)
    print("_______________________________")

    results = pd.DataFrame()

    print("Analysing {0}".format(group))

    X_train = pd.read_csv(data_train_file, index_col=0)
    y_train = pd.read_csv(labels_train_file)[group]
    X_test = pd.read_csv(data_test_file, index_col=0)
    y_test = pd.read_csv(labels_test_file)[group]
    
    if hasattr(snakemake.input, "features"):
        feature_file = snakemake.input["features"]
        feature_selection = snakemake.wildcards.feature_selection
        estimator = snakemake.wildcards.estimator
        features = pd.read_csv(feature_file)["feature"]
        X_train = X_train.loc[:, features]
        X_test = X_test.loc[:, features]
    else:
        feature_selection = "none"
        estimator = "none"

    
    X_train = MinMaxScaler().fit_transform(X_train)
    X_train = sparse.csr_matrix(X_train)

    X_test = MinMaxScaler().fit_transform(X_test)
    X_test = sparse.csr_matrix(X_test)

    print("Train data shape: {}".format(X_train.shape))
    print("Train label shape: {}".format(y_train.shape))
    print("Test data shape: {}".format(X_test.shape))
    print("Test label shape: {}".format(y_test.shape))
    print("_______________________________")
    
    print("Fitting training data")
    start_time = time.time()
    clf, best_parameters = model_fitting(
        group,
        X_train,
        y_train,
        mode,
        method,
        model,
        optimization,
        config_file,
        output_evaluation,
    )
    end_time = time.time()
    dump(clf, output_fitted_model)
    print("_______________________________")

    print("Testing")
    y_pred = clf.predict(X_test)
    print("_______________________________")

    print("Evaluating")

    if mode == "Classification":
        output_evaluation.split("csv")
        result = evaluate_classifier(y_test, y_pred)

    if mode == "MIC":
        result = evaluate_regression(y_test, y_pred)
        output_file_regression_test_values = (
            output_evaluation.rsplit(".")[0] + "_regression_test_values.csv"
        )

        save_regression_test_values(
            y_test, y_pred, output_file_regression_test_values)

    result.insert(0, "Method", method)
    result.insert(1, "Parameters", best_parameters)
    result.insert(2, "Group", group)
    result.insert(3, "Feature_selection", feature_selection)
    result.insert(4, "Estimator", estimator)
    result.insert(5, "Time", end_time - start_time)


    # results = pd.concat([results, result])
    results = results.append(result)
    print("_______________________________")

    print("Saving results to {0}".format(output_evaluation))
    results.to_csv(output_evaluation, index=False)


main()
