from joblib import dump
import argparse
import sys
import os
import utility
import pandas as pd

def get_mode(y_train):
    if all(isinstance(item, str) for item in y_train):
        unique_values = set(y_train)
        if len(unique_values) == 2:
            return "Classification", True
        else:
            return "Classification", False

    elif all(isinstance(item, (int , float)) for item in y_train):
        if all(item == 0 or item == 1 for item in y_train):
            return "Classification", True
        else:
            return "Regression", False

def main(sysargs=sys.argv[1:]):
    print("_______________________________")
    print("Building models")
    print("_______________________________")

    method = snakemake.wildcards.method
    print("________________")
    print(method)

    config_file = utility.config_reader(snakemake.input["conf"])

    model = snakemake.wildcards.model
    print(model)
    current_module = utility.my_import(config_file["Models"][model]["module"])
    dClassifier = getattr(current_module, config_file["Models"][model]["model"])
    dClassifier = dClassifier(**config_file["Models"][model]["params"])

    print(dClassifier)
    dump(dClassifier, snakemake.output[0])


main()
