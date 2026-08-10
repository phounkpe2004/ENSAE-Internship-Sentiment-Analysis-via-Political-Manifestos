# UN Voting Analysis

This project analyses **United Nations General Assembly voting behavior** using statistical models to estimate countries' latent political positions.

The objective is to transform observed voting patterns into quantitative measures of similarity and political positioning between countries.

## Overview

The analysis follows a general pipeline:

```text
Raw UN voting data
        │
        ▼
Data preparation
        │
        ▼
Voting matrix construction
        │
        ▼
Statistical model
        │
        ▼
Bayesian inference
        │
        ▼
Posterior summaries
        │
        ▼
Country positions and comparisons
        │
        ▼
Visualisation
```

The model is estimated using **Bayesian inference**, allowing uncertainty around the estimated country positions to be retained throughout the analysis.

## Data

The `data/` directory contains the datasets used by the analysis.

### Raw data

`data/raw/` contains the original or externally obtained datasets used to construct the voting data.

These include:

* UN General Assembly voting records;
* country codes;
* vote and country matching information;
* additional voting datasets used in the analysis.

### Processed data

`data/processed/` contains datasets generated during the analysis, including posterior summaries such as:

* `BetaSummary.csv`
* `ThetaSummary.csv`

These files are intermediate outputs of the statistical modelling pipeline.

## Method

The core of the project is a probabilistic model of UN voting behavior.

The model uses observed votes to estimate latent parameters representing countries' positions and their relationship to individual resolutions.

The main modelling workflow is divided into several stages:

* **data processing** — preparation and transformation of the voting data;
* **model building** — specification of the Bayesian model;
* **inference** — posterior sampling and estimation;
* **post-processing** — extraction and summarisation of posterior results;
* **visualisation** — graphical representation of estimated positions and model outputs.

The implementation relies primarily on **PyMC** and **ArviZ** for Bayesian modelling and posterior analysis.

## Repository structure

```text
un_voting_analysis/
│
├── main.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
└── src/
    ├── config.py
    ├── data_processing.py
    ├── inference.py
    ├── model_builder.py
    ├── postprocessing.py
    ├── viz.py
    └── README.md
```

## Main components

### `data_processing.py`

Functions for loading, cleaning and transforming the UN voting data into the structures required by the statistical model.

### `model_builder.py`

Defines the Bayesian statistical model used to represent voting behavior.

### `inference.py`

Handles model estimation and Bayesian inference.

### `postprocessing.py`

Processes posterior results and produces summary statistics and derived quantities.

### `viz.py`

Contains functions used to visualise the estimated parameters and country positions.

### `config.py`

Contains configuration parameters used throughout the analysis, including settings for the Bayesian sampling procedure.

## Notebook

The main analysis is contained in:

```text
main.ipynb
```

The notebook provides the main entry point for running the analysis and inspecting its results.

## Installation

From the root of the repository:

```bash
pip install -r requirements.txt
```

The project requires Python and the packages specified in the root-level `requirements.txt`.

## Reproducibility

The statistical analysis relies on Bayesian sampling and may therefore produce slightly different numerical results across runs depending on the sampling configuration and random seeds.

The configuration used for the analysis is defined in the project source code and can be adjusted through `config.py`.

## Data sources

The raw datasets are derived from publicly available UN voting data and also the dataset used by Voeten for his research work.
