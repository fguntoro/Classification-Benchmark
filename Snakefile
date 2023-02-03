configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################


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
            group=config["GROUP"],
            feature_selection=config["FEATURE_SELECTION"]
        )