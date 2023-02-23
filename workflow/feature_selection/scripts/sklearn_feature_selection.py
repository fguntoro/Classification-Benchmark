import argparse
from joblib import load

import numpy as np
import pandas as pd

import sys
sys.path.insert(1, 'workflow/support')
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
        required=False,
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
        required=True,
    )
    parser.add_argument(
        "--estimator",
        dest="estimator",
        help="Saved joblib of Tuned estimator",
        required=False,
    )

    args = parser.parse_args()
    feature_selection = args.feature_selection
    path_data = args.path_data
    output = args.output
    group = args.group
    path_indices = args.indices

    config_file = utility.config_reader(args.config)

    current_module = utility.my_import(
        config_file["Feature_Selection"][feature_selection]["module"])
    Feature_Selector = getattr(
        current_module, config_file["Feature_Selection"][feature_selection]["name"])
    params = config_file["Feature_Selection"][feature_selection]["params"]

    if feature_selection == "VarianceThreshold":
        Feature_Selector = Feature_Selector(**params)
    elif feature_selection in ["sklearn_RFE", "RFECV" , "SelectFromModel" , "SequentialFeatureSelectorForward" , "SequentialFeatureSelectorBackward"]:
        path_label = args.path_label
        estimator = load(args.estimator)
        Feature_Selector = Feature_Selector(estimator, **params)      

    print(Feature_Selector)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('selector', Feature_Selector)])
    
    X = pd.read_csv(path_data, index_col=0)

    for file in path_indices:
        feature_names_remove = pd.read_csv(file)['feature']
        X = X.drop(feature_names_remove, axis =1)

    if feature_selection == "VarianceThreshold":
        pipe.fit(X)
    elif feature_selection in ["sklearn_RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelectorForward", "SequentialFeatureSelectorBackward"]:
        y = pd.read_csv(path_label, index_col=0)[group]
        y = np.ravel(y)
        pipe.fit(X,y)
    
    results = np.delete(
        X.columns, pipe.named_steps['selector'].get_support(indices=True))
    print(X.columns)
    print(Feature_Selector.get_support(indices=True))
    print(results)
    pd.DataFrame({"feature":results}).to_csv(output, index=False)


main()
