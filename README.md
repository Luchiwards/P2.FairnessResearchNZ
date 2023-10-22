# Fairness analysis and ML models New Zealand prosecution and conviction data

This repository contains the source and data files necessary to
reproduce the work presented in a paper submitted as part of COMPSCI
760 at University of Auckland, October, 2023.

## Requirements

### R

To generate some of the plots, you need to have a base R installation.
On Debian (Bookworm as of this writing), this will install the
necessary requirements:

```sh
apt install r-base
```

### Python

See the files `Pipfile` and `requirements.txt` for the necessary
components and install them using your favourite dependency manager.

## Data

The datasets used are in the `data/` subdirectory. See the paper for
source information.

## TeX

The source to the the paper, as well as a Makefile to build it, are in
the `tex/` directory.

To build the paper, along with XXX some XXX of the necessary plots,
install the necessary requirements (a working TeX installation and the
`latexmk` program) and excecute the following:

```sh
cd tex
make
```

## Data analysis plots

To generate the plots used in the data analysis portion, execute the
following from the top level of the repository (the one containing
this README):

```sh
Rscript r/justice-base-rates.R
Rscript r/police-base-rates.R
```

The generated plots will be placed in the current directory.

## Machine learning models and metrics

Our study builds several machine learning models and performs several
analyses on each of them. It stores all the results in the file

```
metrics.json
```

The files to run the mitigations on the datasests can be found in the
following path `/toolkit/analysis`. It takes several minutes to
complete this process. Execute the following:

```sh
python main.py run_mitigations_justice
python main.py run_mitigations_police
```

The script produces different plots that can be found in the following
path: `/toolkit/plots`. The path for the plots will be created during
the run of the scripts above.

A top layer was created to manage the datasets as objects that the
AIF360 toolkit can handle this objects can be found in the following
path: `/toolkit/datasets_objects`. The files that contain the objects
are:

```
justice.py
police.py
```

To generate the metrics run

```sh
python main.py plot_metrics
```

All the models will be saved after being trained inside toolkit/models

### Selecting models and mitigations

Inside toolkit/analysis/run_mitigations_justice.py and toolkit/analysis/run_mitigations_police.py, in the object 
FairnessProcessMitigation the models and mitigations can be set.

models_to_fit contains an array of the models that are available to run.
preprocessing_strategies contains the preprocessing strategies available.
postprocessing_strategy contains the postprocessing strategy available.