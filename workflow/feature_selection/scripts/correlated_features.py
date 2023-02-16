import argparse
import sys
import os
import numpy as np
import pandas as pd

def correlated_features(X, correlation_threshold = 0.90):
    """
    Identifies features that are highly correlated. Let's assume that if
    two features or more are highly correlated, we can randomly select
    one of them and discard the rest without losing much information.
    
    
    Parameters
    ----------
    X : pandas dataframe
        A data set where each row is an observation and each column a feature.
        
    correlation_threshold: float, optional (default = 0.90)
        The threshold used to identify highly correlated features.
        
    Returns
    -------
    labels: list
        A list with the labels identifying the features that contain a 
        large fraction of constant values.
    """
    
    # Make correlation matrix
    corr_matrix = X.corr(method = "spearman").abs()
    
    
    # Select upper triangle of matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    

    # Find index of feature columns with correlation greater than correlation_threshold
    labels = [column for column in upper.columns if any(upper[column] >  correlation_threshold)]
    
    return labels



def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Removing Correlated Features")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_data",
        dest="path_data",
        help="Path to the data file",
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
        "--correlation_threshold",
        dest="correlation_threshold",
        help="Correlation threshold for removing features",
        required=True,
    )

    args = parser.parse_args()
    path_data = args.path_data
    output = args.output
    group = args.group
    path_indices = args.indices
    correlation_threshold = float(args.correlation_threshold)

    X = pd.read_csv(path_data)
    col_idx_names = pd.read_csv(path_indices)['feature']

    X = X.drop(col_idx_names, axis=1)

    results = correlated_features(X, correlation_threshold=correlation_threshold)

    print(results)

    pd.DataFrame({"feature":results}).to_csv(output, index=False)


main()
