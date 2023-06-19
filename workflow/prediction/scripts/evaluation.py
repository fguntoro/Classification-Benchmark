import sys
import numpy as np
import time
import pandas as pd
from scipy import sparse

# Input and output file paths
data_train_file = snakemake.input["data_train"]
labels_train_file = snakemake.input["label_train"]
data_test_file = snakemake.input["data_test"]
labels_test_file = snakemake.input["label_test"]
model = snakemake.input["model"]
modelname = snakemake.wildcards.model
method_config_file = snakemake.input["conf"]
group = snakemake.wildcards.group
data = snakemake.wildcards.data

# Optimization configuration
optimization = snakemake.config["OPTIMIZATION"]

# Output evaluation file path
output_evaluation = snakemake.output["evaluation"]

# Function to determine the mode of the target variable
def get_mode(y_train):
    if all(isinstance(item, str) for item in y_train):
        unique_values = set(y_train)
        if len(unique_values) == 2:
            return "Classification", True
        else:
            return "Classification", False
        
    elif all(isinstance(item, (int, float)) for item in y_train):
        if all(item == 0 or item == 1 for item in y_train):
            return "Classification", True
        else:
            return "Regression", False

# Main function
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

    # Read the configuration file
    config_file = utility.config_reader(method_config_file)

    # DataFrame to store results
    results = pd.DataFrame()

    print("Analysing {0}".format(group))

    # Read data and labels
    X_train = pd.read_csv(data_train_file, index_col=0)
    y_train = pd.read_csv(labels_train_file)[group]
    X_test = pd.read_csv(data_test_file, index_col=0)
    y_test = pd.read_csv(labels_test_file)[group]

    # Determine the mode of the target variable
    mode, isBinary = get_mode(y_train)

    print("Mode = ", mode)
    print("Model = ", model)
    print("Config file = ", method_config_file)
    print("Optimization = ", optimization)
    print("_______________________________")

    # Feature selection and estimator settings
    if hasattr(snakemake.input, "features"):
        feature_file = snakemake.input["features"]
        features = pd.read_csv(feature_file)["feature"]
        print(features)
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

    # Data preprocessing
    n_features = len(X_test.columns)
    print(n_features)

    print("Train data shape: {}".format(X_train.shape))
    print("Train label shape: {}".format(y_train.shape))
    print("Test data shape: {}".format(X_test.shape))
    print("Test label shape: {}".format(y_test.shape))
    print("_______________________________")

    start_time = time.time()

    if n_features != 0:
        # Model fitting
        print("Fitting training data")
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
        print("_______________________________")

        # Model testing
        print("Testing")
        y_pred = clf.predict(X_test)
        print("_______________________________")

        # Model evaluation
        print("Evaluating")
        if mode == "Classification":
            if hasattr(clf, "best_estimator_"):
                best_model = clf.best_estimator_.fit(X_train, y_train)
            else:
                best_model = clf.fit(X_train, y_train)
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
    else:
        result = pd.DataFrame([dict(fail=1)])
        best_parameters = "None"

    end_time = time.time()

    # Update the result DataFrame
    result.insert(0, "Data", data)
    result.insert(1, "Group", group)
    result.insert(2, "Method", modelname)
    result.insert(3, "Parameters", best_parameters)
    result.insert(4, "Feature_selection", feature_selection)
    result.insert(5, "Estimator", estimator)
    result.insert(6, "N_feature", n_features)
    result.insert(7, "Time", end_time - start_time)

    # Append the result to the results DataFrame
    results = results.append(result)
    print("_______________________________")

    print("Saving results to {0}".format(output_evaluation))
    results.to_csv(output_evaluation, index=False)


main()
