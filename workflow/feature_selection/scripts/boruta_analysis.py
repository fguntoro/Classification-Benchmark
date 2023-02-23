import argparse
import os
from joblib import load
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, 'workflow/support')
import utility

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from boruta import BorutaPy


def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Running Feature Selection")

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
        "--output",
        dest="output",
        help="Filename of output with full directory path",
        required=True,
    )
    parser.add_argument(
        "--feature_selection",
        dest="feature_selection",
        help="Method selected for feature selection",
        required=False,
    )
    parser.add_argument(
        "--estimator",
        dest="estimator",
        help="Saved joblib of Tuned estimator",
        required=True,
    )

    args = parser.parse_args()
    feature_selection = args.feature_selection
    path_data = args.path_data
    path_label = args.path_label
    output = args.output
    group = args.group
    path_indices = args.indices
    estimator = load(args.estimator)


    ## LOAD DATA
    X = pd.read_csv(path_data, index_col=0)

    for file in path_indices:
        feature_names_remove = pd.read_csv(file)['feature']
        X = X.drop(feature_names_remove, axis =1)

    Xval = X.values
    y = pd.read_csv(path_label, index_col=0)
    y = np.ravel(y)

    config_file = utility.config_reader(args.config)

    # define Boruta feature selection method
    Feature_Selector = BorutaPy(estimator, n_estimators='auto', verbose=2, random_state=12345)

    print(Feature_Selector)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('selector', Feature_Selector)])
    
    
    pipe.fit(Xval,y)

    results =  X.columns[Feature_Selector.support_]
    pd.DataFrame({"feature":results}).to_csv(output, index=False)

main()