from joblib import dump
import argparse
import sys
import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(1, 'workflow/support')
from support import (
    preprocess,
)
import utility
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


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
        help="Path to file containing vector of index for training",
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
        required=True,
    )

    args = parser.parse_args()
    feature_selection = args.feature_selection
    path_data = args.path_data
    path_label = args.path_label
    output = args.output
    group = args.group
    path_indices = args.indices

    config_file = utility.config_reader(args.config)

    current_module = utility.my_import(
        config_file["Feature_Selection"][feature_selection]["module"])
    Feature_Selector = getattr(
        current_module, config_file["Feature_Selection"][feature_selection]["name"])
    params = config_file["Feature_Selection"][feature_selection]["params"]

    if feature_selection in ["RFE", "RFECV" , "SelectFromModel" , "SequentialFeatureSelectorForward" , "SequentialFeatureSelectorBackward"]:
        estimator_name = config_file["estimator"]
        models_config_file = utility.config_reader("workflow/prediction/rules/models_config.yml")
        current_module = utility.my_import(models_config_file["Models"][estimator_name]["module"])
        estimator = getattr(current_module, models_config_file["Models"][estimator_name]["model"])
        estimator = estimator(**models_config_file["Models"][estimator_name]["params"])
        Feature_Selector = Feature_Selector(estimator, **params)        
    else:
        Feature_Selector = Feature_Selector(**params)

    print(Feature_Selector)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('selector', Feature_Selector)])
    
    X = pd.read_csv(path_data)
    y = pd.read_csv(path_label)

    if feature_selection == "VarianceThreshold":
        pipe.fit(X)
    else:
        pipe.fit(X,y)

    results =  X.columns[Feature_Selector.get_support(indices=True)]
    pd.DataFrame({"feature":results}).to_csv(output, index=False)


main()
