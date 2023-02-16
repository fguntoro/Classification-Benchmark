configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################

include: "workflow/preprocessing/rules/snakefile"
include: "workflow/feature_selection/rules/snakefile"
include: "workflow/prediction/rules/snakefile"


########################
# One to rule them all #
########################


rule all:
    input:
        expand(
            #config["OUTPUT_DIR"] + "/{group}/prediction/summary.csv",
            config["OUTPUT_DIR"] + "/{group}/feature_selection/{feature_selection}/{feature_selection}.csv",
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/variance_threshold/variance_threshold_indices.csv",
            #config["OUTPUT_DIR"] + "/{group}/feature_selection/correlated_features/correlated_features_indices.csv",
            #config["OUTPUT_DIR"] + "/{group}/prediction/results/{feature_selection}/{method}.csv",
            #config["OUTPUT_DIR"] + "/{group}/prediction/results/{method}.csv",
            #config["OUTPUT_DIR"] + "/{group}/preprocessing/split_data/X_train.csv",
            group=config["GROUP"],
            method=config["METHODS"],
            feature_selection=config["FEATURE_SELECTION"]
        )