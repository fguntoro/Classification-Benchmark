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
modelname = snakemake.wildcards.model
method_config_file = snakemake.input["conf"]
group = snakemake.wildcards.group

optimization = snakemake.config["OPTIMIZATION"]

output_evaluation = snakemake.output["evaluation"]
output_fitted_model = snakemake.output["fitted_model"]


def get_mode(y_train):
    if all(isinstance(item, str) for item in y_train):
        unique_values = set(y_train)
        if len(unique_values) == 2:
            return "Classification", True
        else:
            return "Classification", False
        
    elif all(isinstance(item, int) for item in y_train):
        if all(item == 0 or item == 1 for item in y_train):
            return "Classification", True
        else:
            return "Regression", False

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
    import shap
    import matplotlib.pyplot as plt


    #label.set_index(label.columns[0], inplace=True, drop=True)
    config_file = utility.config_reader(method_config_file)



    results = pd.DataFrame()

    print("Analysing {0}".format(group))

    X_train = pd.read_csv(data_train_file, index_col=0)
    y_train = pd.read_csv(labels_train_file)[group]
    X_test = pd.read_csv(data_test_file, index_col=0)
    y_test = pd.read_csv(labels_test_file)[group]


    mode, isBinary = get_mode(y_train)

    print("Mode = ", mode)
    print("Model = ", model)
    print("Config file = ", method_config_file)
    print("Optimization = ", optimization)
    print("_______________________________")

    if hasattr(snakemake.input, "features"):
        feature_file = snakemake.input["features"]
        features = pd.read_csv(feature_file)["feature"]
        X_train = X_train.loc[:, features]
        X_test = X_test.loc[:, features]

        try:
            feature_selection = snakemake.wildcards.feature_selection
            estimator = snakemake.wildcards.estimator
        except AttributeError:
            feature_selection = "stability"
            estimator = "none"
        
    else:
        feature_selection = "none"
        estimator = "none"

    
    #X_train = MinMaxScaler().fit_transform(X_train)
    #X_train = sparse.csr_matrix(X_train)

    n_features = len(X_test.columns)
    #X_test = MinMaxScaler().fit_transform(X_test)
    #X_test = sparse.csr_matrix(X_test)

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
        modelname,
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
        best_model = clf.best_estimator_.fit(X_train, y_train)
        explainer = shap.Explainer(best_model, X_train)
        shap_test = explainer(X_test)

        if modelname == "RFC":
            filename = output_evaluation

            filename_shap_bar = filename.replace(".csv", "_shap_bar.png")
            shap.plots.bar(shap_test[:, :, 1], show=False)
            print("Saving SHAP plot")
            plt.savefig(filename_shap_bar)

            filename_shap_summary = filename.replace(".csv", "_shap_summary.png")
            shap.summary_plot(shap_test[:, :, 1], show=False)
            plt.savefig(filename_shap_summary)

            filename_shap_heatmap = filename.replace(".csv", "_shap_heatmap.png")
            shap.plots.heatmap(shap_test[:, :, 1], show=False)
            plt.savefig(filename_shap_heatmap)

        if modelname == "LR_elasticnet":
            filename = output_evaluation

            filename_shap_bar = filename.replace(".csv", "_shap_bar.png")
            shap.plots.bar(shap_test, show=False)
            print("Saving SHAP plot")
            plt.savefig(filename_shap_bar)

            filename_shap_summary = filename.replace(".csv", "_shap_summary.png")
            shap.summary_plot(shap_test, show=False)
            plt.savefig(filename_shap_summary)

            filename_shap_heatmap = filename.replace(".csv", "_shap_heatmap.png")
            shap.plots.heatmap(shap_test, show=False)
            plt.savefig(filename_shap_heatmap)

    if mode == "Classification":
        output_evaluation.split("csv")
        result = evaluate_classifier(y_test, y_pred)

    elif mode == "Regression":
        result = evaluate_regression(y_test, y_pred)
        output_file_regression_test_values = (
            output_evaluation.rsplit(".")[0] + "_regression_test_values.csv"
        )

        save_regression_test_values(
            y_test, y_pred, output_file_regression_test_values)

    result.insert(0, "Method", modelname)
    result.insert(1, "Parameters", best_parameters)
    result.insert(2, "Group", group)
    result.insert(3, "Feature_selection", feature_selection)
    result.insert(4, "Estimator", estimator)
    result.insert(6, "N_feature", n_features)
    result.insert(5, "Time", end_time - start_time)


    # results = pd.concat([results, result])
    results = results.append(result)
    print("_______________________________")

    print("Saving results to {0}".format(output_evaluation))
    results.to_csv(output_evaluation, index=False)


main()
