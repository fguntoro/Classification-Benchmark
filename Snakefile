configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################

include: "workflow/preprocessing/rules/snakefile"
include: "workflow/feature_selection/rules/snakefile"
include: "workflow/prediction/rules/snakefile"

import pandas as pd

def get_groups(config):
    if config["GROUP"] == "all":
        df = pd.read_csv(config["FILE_LABEL"], index_col = 0)
        groups = df.columns.tolist()
    else:
        groups = config["GROUP"]
    return groups



########################
# One to rule them all #
########################

rule all:
    input:
        expand(
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/estimator_tuning/tuned_RFC.joblib",
            config["OUTPUT_DIR"] + "/summary.csv",
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/{feature_selection}/{feature_selection}_RFC.csv",
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/variance_threshold/variance_threshold_indices.csv",
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/correlated_features/correlated_features_indices.csv",
            #config["OUTPUT_DIR"] + "/{group}/prediction/results/{feature_selection}/{method}.csv",
            #config["OUTPUT_DIR"] + "/{group}/prediction/results/{method}.csv",
            #config["OUTPUT_DIR"] + "/{data}/{group}/preprocessing/split_data/X_train.csv",
            #config["OUTPUT_DIR"] + \
            #"/{data}/{group}/prediction/results/{method}/LR_l1-nofeatureselection/summary.csv",

            #data=config["FILE_DATA"],
            #group=get_groups(config),
            #method=config["METHODS"],
            #feature_selection=config["FEATURE_SELECTION"]
        )

