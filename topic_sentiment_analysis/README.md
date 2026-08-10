# Topic Sentiment Analysis

This repository contains the data, code, and notebooks used for the computational analysis of political party manifestos.

The project focuses on political manifestos from **Australia and the United States** and explores political texts using several computational text-analysis approaches, including sentiment analysis, topic analysis, and WordScores.

## Project structure

```text
.
├── data/
│   ├── raw/             # Original manifesto texts
│   ├── intermediate/    # Data generated during intermediate processing
│   └── processed/       # Processed and structured datasets
│
├── notebooks/           # Jupyter notebooks for exploration and analysis
│
├── src/                 # Python source code
│   ├── sentiment_analysis/
│   ├── topic_analysis/
│   └── wordscores/


```

## Data

The repository contains political manifesto data from:

* **Australia**, covering multiple federal elections and political parties;
* **United States**, covering presidential elections from 1960 to 2024.

The original manifesto texts are stored in:

```text
data/raw/manifestos/
```

with separate directories for Australia and the United States.

Processed and intermediate datasets are stored in:

```text
data/processed/
data/intermediate/
```

See [`data/README.md`](data/README.md) for additional information about the organization of the data.

## Methods

The project currently includes several computational approaches to political text analysis.

### Sentiment analysis

The `src/sentiment_analysis/` directory contains the code used for sentiment analysis of manifesto texts, including the extraction and analysis of foreign-affairs-related content.

### Topic analysis

The `src/topic_analysis/` directory contains preprocessing utilities and functions used for topic analysis, including labelled/seeded topic modelling and word-distribution analysis.

### WordScores

The `src/wordscores/` directory contains the material related to the WordScores analysis.

## Notebooks

The `notebooks/` directory contains the Jupyter notebooks used for data loading, exploratory analysis, preprocessing, modelling, and analysis.

These notebooks document the different stages of the research workflow and can be used to reproduce or extend parts of the analysis.

## Installation

Clone the repository and install the required Python packages:

    pip install -r requirements.txt

Download the spaCy English language model:

    python -m spacy download en_core_web_sm

## Reproducibility

The repository separates:

1. raw source data;
2. intermediate datasets;
3. processed datasets;
4. analysis code;
5. Jupyter notebooks.

This organization is intended to make the research workflow easier to understand, reproduce, and extend.

## Status

This project is under development.

## License

License information will be added once the appropriate licensing terms for the code and data have been determined.
