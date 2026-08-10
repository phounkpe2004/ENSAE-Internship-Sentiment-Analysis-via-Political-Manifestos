# Notebooks

This directory contains the Jupyter notebooks used throughout the project for data loading, exploration, preprocessing, modelling, and analysis.

## Notebooks

| Notebook                      | Purpose                                      |
| ----------------------------- | -------------------------------------------- |
| `manifesto_file_loader.ipynb` | Loading and preparing manifesto data         |
| `sentiment_analyzer.ipynb`    | Sentiment analysis of manifesto texts        |
| `exploratory_analysis.ipynb`  | Exploratory analysis of the manifesto corpus |
| `seed_words_extractor.ipynb`  | Extraction and exploration of seed words     |
| `topic_analysis.ipynb`        | Topic modelling and analysis                 |
| `wordcodes.ipynb`             | WordScores analysis                          |

## Relationship with the source code

The notebooks use the reusable Python code located in [`../src/`](../src/).

They also read from and, where applicable, generate data in [`../data/`](../data/).

The notebooks are primarily intended for exploration, analysis, visualization, and documenting the research workflow, while reusable functions and processing logic are kept in `src/`.
