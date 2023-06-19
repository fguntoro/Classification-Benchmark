# Classification Benchmark

Snakemake pipeline for multi-omics data analysis. Current usage to train and evaluate machine learning prediction models.


## Installation

Clone this repo and set up an environment containing [Snakemake](https://snakemake.readthedocs.io/en/stable/) 

```bash
git clone https://github.com/fguntoro/Classification-Benchmark.git
conda activate base
mamba create -c conda-forge -c bioconda -n snakemake snakemake
```

Make sure to install any dependency e.g. python and R packages. For example:
```bash
mamba env update -n snakemake -f ./Classification-Benchmark/workflow/prediction/envs/env.yml
```

## Usage - example
If running locally:
```bash
snakemake -s Snakefile --use-conda
```

On the cluster:
```bash
bash snakemake_job.sh > snakemake.log
```

## Usage
To run the pipeline using your own data, you can update the config.yaml file:
```python
INPUT_DIR: "example_data"

FILE_DATA: ["madelon_like_X"]

FILE_LABEL: "example_data/madelon_like_y.csv"

GROUP: "all"

OUTPUT_DIR: "output"

FEATURE_SELECTION: ["stability", "boruta", "sklearn_RFE"]

ESTIMATOR: ["RF"]

METHODS: ["linear_model"]

OPTIMIZATION: "GridSearchCV"
```

