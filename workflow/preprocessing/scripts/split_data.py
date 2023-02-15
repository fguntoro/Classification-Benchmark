import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

import sys
sys.path.insert(1, 'workflow/support')
import support

def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Splitting data")

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
        "--config",
        dest="config",
        help="Path to the config file",
        required=False,
    )
    parser.add_argument(
        "--group",
        dest="group",
        help="Label group/column for directory",
        required=True,
    )
    parser.add_argument(
        "--output",
        nargs='+',
        default=[],
        dest="output",
        help="List of output files",
        required=True,
    )
    
    args = parser.parse_args()
    path_data = args.path_data
    path_label = args.path_label
    group = args.group
    output = args.output

    print("Data file = ", path_data)
    print("Label file = ", path_label)
    print("_______________________________")
    data, labels = support.preprocess(data=path_data, label=path_label)

    y = np.ravel(labels[[group]])
    na_index = np.isnan(y)
    y = y[~na_index]

    X = data
    X = X.loc[~na_index,:]

    print("Data shape: {}".format(X.shape))
    print("Label shape: {}".format(y.shape))
    print("_______________________________")

    print("Train-Test split")
    # Make sure that stratified split is disabled for regression models

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # if mode == "Classification":
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, **config_file["TrainTestSplit"]
    #     )

    # if mode == "MIC":
    #     dict_train_test_split = config_file["TrainTestSplit"]
    #     dict_train_test_split["stratify"] = None
    #     print(dict_train_test_split)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, **dict_train_test_split
    #     )

    print("Train data length: {}".format(len(y_train)))
    print("Test data length: {}".format(len(y_test)))
    
    print("_______________________________")
    print("Saving data")

    print("_______________________________")

    pd.DataFrame(X_train).to_csv(output[0], index=False)
    pd.DataFrame(y_train).to_csv(output[1], index=False)
    pd.DataFrame(X_test).to_csv(output[2], index=False)
    pd.DataFrame(y_test).to_csv(output[3], index=False)

main()