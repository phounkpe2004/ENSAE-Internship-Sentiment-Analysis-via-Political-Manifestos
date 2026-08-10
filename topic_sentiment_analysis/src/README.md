# Source Code

This directory contains the Python source code used for the computational analysis of political manifestos.

## Organization

```text
src/
├── sentiment_analysis/
├── topic_analysis/
└── wordscores/
```

### `sentiment_analysis/`

Contains the code used for sentiment analysis of manifesto texts.

This includes utilities for generating country aliases and the main sentiment-analysis pipeline.

### `topic_analysis/`

Contains the preprocessing and modelling utilities used for topic analysis.

The code includes functions for:

* manifesto text preprocessing;
* seed-word handling;
* labelled / seeded topic modelling;
* word-distribution analysis.

### `wordscores/`

Contains the code and notebooks related to the WordScores analysis.

## Relationship with notebooks

The notebooks used to run, explore, and visualize the analyses are stored separately in `notebooks/`.

The general organization is therefore:

```text
notebooks/
      │
      │ uses
      ▼
src/
      │
      │ reads / writes
      ▼
data/
```

This separation keeps reusable analysis code distinct from exploratory and research notebooks.
