# AutoML-AdvancedML

Course project for Advanced ML focused on creating a custom AutoML algorithm and comparing it with TPOT and Autoskl.

## Project structure

```
AutoML-AdvancedML/
├─ data/
│   ├─ raw/
│   ├─ processed/
│   └─ results/
├─ notebooks/
│   ├─ compare_results.ipynb
│   └─ preprocessing.ipynb
├─ src/
│   ├─ evaluation.py
│   └─ models.py
├─ automl.py
├─ tpot_run.py
├─ autosklearn_run.py
├─ requirements.txt
└─ README.md
```

## Core idea of `automl.py`

`automl.py` runs a custom AutoML search that treats model selection and hyperparameter tuning as one optimization process. The main loop explores different model/hyperparameter states, accepts better candidates, and occasionally accepts worse ones early on to avoid getting stuck, then becomes more conservative over time.

## Setup

Install dependencies with Conda:

```bash
conda create -n automl-advancedml python=3.11 -y
conda activate automl-advancedml
conda install -c conda-forge numpy pandas scikit-learn tpot dask distributed matplotlib jupyter notebook ipykernel
```

In Ubuntu (or WSL Ubuntu), install `auto-sklearn` in the same environment:

```bash
conda install -c conda-forge auto-sklearn
```

Important: `auto-sklearn` must be installed and executed on Linux (Ubuntu). If you are on Windows, run `autosklearn_run.py` from WSL Ubuntu or a Linux machine.

## How to run

Use any dataset in `data/processed`.

```bash
python automl.py <dataset_name.csv> -e 3600
python tpot_run.py <dataset_name.csv> -e 3600
python autosklearn_run.py <dataset_name.csv> -e 3600
```

`-e` means execution time in seconds (search budget). Example: `-e 1800` runs for about 30 minutes.

For all other arguments, use the built-in help:

```bash
python automl.py -h
python tpot_run.py -h
python autosklearn_run.py -h
```





