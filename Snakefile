configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################


include: "workflow/rules/prediction/snakefile"


########################
# One to rule them all #
########################


rule all:
    input:
        expand(
            config["OUTPUT_DIR"] + "/{group}/prediction/summary.csv",
            group=config["GROUP"],
        )