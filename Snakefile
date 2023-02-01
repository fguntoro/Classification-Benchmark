configfile: "config.yaml"


##################################
# Import other Snakemake Modules #
##################################


include: "workflow/prediction/rules/snakefile"


########################
# One to rule them all #
########################


rule all:
    input:
        expand(
            config["OUTPUT_DIR"] + "/{group}/prediction/summary.csv",
            group=config["GROUP"],
        )