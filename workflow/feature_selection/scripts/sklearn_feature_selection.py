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

    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_data",
        dest="path_data",
        help="Path to the data file",
        required=True,
    )
    parser.add_argument(
        "--file_label",
        dest="file_label",
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

    # Read the configuration file
    config_file = utility.config_reader(args.config)

    # Import the Feature_Selector class based on the selected method
    current_module = utility.my_import(
        config_file["Feature_Selection"][feature_selection]["module"])
    Feature_Selector = getattr(
        current_module, config_file["Feature_Selection"][feature_selection]["name"])
    params = config_file["Feature_Selection"][feature_selection]["params"]

    if feature_selection == "VarianceThreshold":
        Feature_Selector = Feature_Selector(**params)
    elif feature_selection in ["sklearn_RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelectorForward", "SequentialFeatureSelectorBackward"]:
        file_label = args.file_label
        estimator = load(args.estimator)
        Feature_Selector = Feature_Selector(estimator, **params)      

    print(Feature_Selector)

    # Create a pipeline with a scaler and the feature selector
    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('selector', Feature_Selector)])
    
    # Load the input data
    X = pd.read_csv(path_data, index_col=0)

    for file in path_indices:
        feature_names_remove = pd.read_csv(file)['feature']
        # Drop the features from DataFrame X if they exist
        existing_features = set(X.columns)
        features_to_drop = list(filter(lambda f: f in existing_features, feature_names_remove))
        X = X.drop(features_to_drop, axis=1)

    if feature_selection == "VarianceThreshold":
        pipe.fit(X)
    elif feature_selection in ["sklearn_RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelectorForward", "SequentialFeatureSelectorBackward"]:
        # Load the labels for supervised feature selection methods
        y = pd.read_csv(file_label, index_col=0)[group]
        y = np.ravel(y)
        pipe.fit(X, y)
    
    # Get the features that were not selected
    results = np.delete(
        X.columns, pipe.named_steps['selector'].get_support(indices=True))
    print(X.columns)
    print(Feature_Selector.get_support(indices=True))
    print(results)
    
    # Save the results to the output file
    pd.DataFrame({"feature": results}).to_csv(output, index=False)


main()
