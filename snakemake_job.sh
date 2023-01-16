#!/bin/bash

module load anaconda3/personal
source activate mres-project2-snakemake

cd $HOME/Project2/BenchmarkDR
snakemake --unlock
snakemake -s Snakefile --rerun-incomplete --verbose --jobs 50 --cluster "qsub -V -lwalltime=24:00:00 -lselect=1:ncpus=8:mem=100gb"

conda deactivate
