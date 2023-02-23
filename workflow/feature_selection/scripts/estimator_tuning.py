import argparse
import sys
import os
from joblib import dump

import numpy as np
import pandas as pd

import sys
sys.path.insert(1, 'workflow/support')
import utility
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Tuning estimator")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_data",
        dest="path_data",
        help="Path to the data file",
        required=True,
    )
    parser.add_argument(
        "--path_label",
        dest="path_label",
        help="Path to the label file",
        required=True,
    )
    parser.add_argument(
        "--indices",
        dest="indices",
        nargs='+',
        default=[],
        help="List of files containing vector of feature names to remove",
        required=False,
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Path to the config file",
        required=True,
    )
    parser.add_argument(
        "--group",
        dest="group",
        help="Label group/column for directory",
        required=True,
    )
    parser.add_argument(
        "--estimator_name",
        dest="estimator_name",
        help="Estimator to be tuned",
        required=True,
    )
    parser.add_argument(
        "--output",
        dest="output",
        help="Filename of output with full directory path",
        required=True,
    )

    args = parser.parse_args()
    path_data = args.path_data
    path_label = args.path_label
    group = args.group
    estimator_name = args.estimator_name
    output = args.output
    path_indices = args.indices


    ### Load Data
    X = pd.read_csv(path_data)

    for file in path_indices:
        feature_names_remove = pd.read_csv(file)['feature']
        X = X.drop(feature_names_remove, axis =1)
    y = pd.read_csv(path_label)
    y = np.ravel(y[group])

    ### Load Estimator for tuning from config
    config_file = utility.config_reader(args.config)

    current_module = utility.my_import(config_file["Estimator"][estimator_name]["module"])
    estimator = getattr(current_module, config_file["Estimator"][estimator_name]["model"])
    estimator = estimator(**config_file["Estimator"][estimator_name]["params"])

    print(estimator)

    # Define parameter grid
    param_grid_tmp = config_file["Estimator"][estimator_name]["param_grid"]

    param_grid = {"estimator__" + str(k): v for k, v in param_grid_tmp.items()}
    for key, value in param_grid.items():
        if isinstance(value, str):
            param_grid[key] = eval(param_grid[key])
    param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}
    print(param_grid)


    pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('estimator', estimator)])

    # Initialize GridSearch object
    gscv = GridSearchCV(pipe, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = 'roc_auc')

    # Fit gscv
    gscv.fit(X, y)

    # Get best parameters and score
    best_params_tmp = gscv.best_params_
    best_params = {k.replace("estimator__", ""): v for k, v in best_params_tmp.items()}

    best_score = gscv.best_score_
    print(best_params)

    # Update classifier parameters
    estimator.set_params(**best_params)

    print(estimator)
    dump(estimator, output)

main()
