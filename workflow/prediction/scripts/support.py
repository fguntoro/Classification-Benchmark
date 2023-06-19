import utility
import pandas as pd
import numpy as np
from joblib import load, dump


def CVFolds(config_file):
    """
    Creates cross-validation folds based on the specified splitter class and parameters.

    Args:
        config_file (dict): Configuration file containing the splitter class information.

    Returns:
        object: Cross-validation folds object.
    """
    module = utility.my_import(config_file["SplitterClass"]["module"])
    splitter = getattr(module, config_file["SplitterClass"]["splitter"])
    folds = splitter(**config_file["SplitterClass"]["params"])
    return folds


def preprocess(data, label):
    """
    Preprocesses the data and label files by merging them and separating the label columns.

    Args:
        data (str): Path to the data file.
        label (str): Path to the label file.

    Returns:
        tuple: Preprocessed data and labels.
    """
    label = pd.read_csv(label)
    label.set_index(label.columns[0], inplace=True, drop=True)

    data = pd.read_csv(data)
    data.set_index(data.columns[0], inplace=True, drop=True)
    data = data.merge(label, left_index=True, right_index=True)

    labels = pd.DataFrame()

    for col in label.columns:
        labels = pd.concat([labels, data[[col]]], axis=1)
        data = data.drop([col], axis=1)

    print("label set: {0}".format(labels.shape))
    print("data set: {0}".format(data.shape))
    print("_______________________________")

    return (data, labels)


def model_fitting(
    col,
    X_train,
    y_train,
    mode,
    modelname,
    model_joblib,
    optimization,
    config_file,
    output_file,
):
    """
    Fits a model to the training data and performs hyper-parameter tuning if specified.

    Args:
        col (str): Name of the column being predicted.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        mode (str): Mode of the model ('Classification' or 'Regression').
        modelname (str): Name of the model.
        model_joblib (str): Path to the pre-trained model file.
        optimization (str): Type of hyper-parameter optimization ('GridSearchCV', 'RandomizedSearchCV', or 'None').
        config_file (dict): Configuration file containing model and cross-validation information.
        output_file (str): Path to the output file.

    Returns:
        tuple: Fitted model and best parameters (if hyper-parameter tuning is performed).
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import (
        balanced_accuracy_score,
        make_scorer,
        f1_score,
        roc_auc_score,
        mean_squared_error,
        r2_score,
    )

    print("Loading ", model_joblib)
    model = load(model_joblib)
    modelname = modelname

    if mode == "Classification":
        scoring = dict(
            Accuracy="accuracy",
            tp=make_scorer(utility.tp),
            tn=make_scorer(utility.tn),
            fp=make_scorer(utility.fp),
            fn=make_scorer(utility.fn),
            balanced_accuracy=make_scorer(balanced_accuracy_score),
            # f1score=make_scorer(f1_score),
            # roc_auc=make_scorer(roc_auc_score),
        )
        refit_metric = "balanced_accuracy"

    if mode == "Regression":
        scoring = dict(
            mean_squared_error="neg_mean_squared_error",
            # mean_squared_log_error ="neg_mean_squared_log_error",
            r2_score="r2",
        )
        refit_metric = "mean_squared_error"

    if optimization != "None":
        print("Hyper-parameter Tuning")

        # The following statement is necessary for methods without parameters to tune, e.g. Linear Regression
        if config_file["Models"][modelname].get("cv") is None:
            print("Not running hyper-parameter tuning")
            clf = model.fit(X_train, y_train)
            print("_______________________________")

            return clf, str(model.get_params())

        param_grid = config_file["Models"][modelname]["cv"]

        for key, value in param_grid.items():
            if isinstance(value, str):
                param_grid[key] = eval(param_grid[key])
        param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}
        print(param_grid)

        for i in config_file["CrossValidation"].items():
            print("{}: {}".format(i[0], i[1]))

        if optimization == "GridSearchCV":
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=CVFolds(config_file),
                refit=refit_metric,
                # refit=utility.refit_strategy,
                **config_file["CrossValidation"]
            )

        elif optimization == "RandomizedSearchCV":
            grid = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                scoring=scoring,
                cv=CVFolds(config_file),
                refit=refit_metric,
                # refit= utility.refit_strategy,
                **config_file["CrossValidation"]
            )

        grid.fit(X_train, y_train)
        print("Best params: {}".format(grid.best_params_))

        # filename = model_joblib
        # filename = filename.replace(".joblib", "_grid.joblib")
        # print("Saving grid model to {0}".format(filename))
        # dump(grid.best_estimator_, filename)

        filename = output_file
        filename = filename.replace(".csv", "_cv.csv")
        print("Saving cv results to {0}".format(filename))
        cv_results = pd.DataFrame(grid.cv_results_)
        cv_results.to_csv(filename, index=False)
        print("_______________________________")

        return grid, str(grid.best_params_)

    elif optimization == "None":
        print("Not running hyper-parameter tuning")
        clf = model.fit(X_train, y_train)
        print("_______________________________")

        return clf, str(model.get_params())


def evaluate_classifier(y_test, y_pred):
    """
    Evaluates the performance of a classification model by calculating various metrics.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        DataFrame: Evaluation results.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        roc_auc_score,
        confusion_matrix,
        f1_score,
    )

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    # f1score = f1_score(y_test, y_pred, pos_label="yes")
    # roc_auc = roc_auc_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    result = dict(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        # f1_score=f1score,
        # roc_auc=roc_auc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        sensitivity=sensitivity,
        specificity=specificity,
    )
    result = pd.DataFrame([result])

    return result


def save_regression_test_values(y_test, y_pred, filename):
    """
    Saves the true and predicted values of a regression model to a file.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        filename (str): Path to the output file.
    """
    dict_values = {"y_test": y_test, "y_predicted": y_pred}
    df_values = pd.DataFrame.from_dict(dict_values)
    df_values.to_csv(filename, index=False)


def evaluate_regression(y_test, y_pred):
    """
    Evaluates the performance of a regression model by calculating various metrics.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        DataFrame: Evaluation results.
    """
    from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

    mse = mean_squared_error(y_test, y_pred)
    # msle = mean_squared_log_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result = dict(mean_squared_error=mse, r2_score=r2)
    result = pd.DataFrame([result])

    return result
