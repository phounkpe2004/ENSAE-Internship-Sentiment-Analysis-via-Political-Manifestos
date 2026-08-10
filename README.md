# Political Text and UN Voting Analysis

This repository contains two complementary research projects studying political positions through **political manifestos** and **United Nations voting behavior**.

The projects approach political positioning from two different but related sources of evidence:

1. **Political manifestos** — analysis of the policy content and sentiment expressed in party manifestos.
2. **UN voting behavior** — estimation and analysis of countries' positions based on their voting patterns in the United Nations General Assembly.

Together, these projects provide complementary perspectives on political preferences and foreign policy positioning.

## Repository structure

```text
.
├── topic_sentiment_analysis/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   └── README.md
│
├── un_voting_analysis/
│   ├── data/
│   ├── src/
│   ├── main.ipynb
│   └── README.md
│
├── requirements.txt
└── README.md
```

## Projects

### Topic and sentiment analysis

The `topic_sentiment_analysis` project analyses political manifestos from the **United States and Australia**.

The pipeline combines:

* manifesto corpus construction and preprocessing;
* seeded topic modelling using LLDA;
* extraction of topic-specific manifesto sentences;
* sentiment analysis;
* identification of countries mentioned in foreign-affairs statements.

The current analysis focuses particularly on the **Foreign Affairs** policy dimension.

See [`topic_sentiment_analysis/README.md`](topic_sentiment_analysis/README.md) for details.

### UN voting analysis

The `un_voting_analysis` project analyses voting behavior in the **United Nations General Assembly**.

The project uses statistical modelling to estimate latent positions from countries' voting patterns. The workflow includes data preparation, model construction, Bayesian inference, post-processing and visualization.

See [`un_voting_analysis/README.md`](un_voting_analysis/README.md) for details.

## Relationship between the projects

The two projects are designed to be complementary.

The manifesto analysis provides information about **what political parties say**, with a particular focus on foreign affairs and the sentiment associated with references to other countries.

The UN voting analysis provides information about **how countries vote in an international institutional setting**.

Combining these perspectives makes it possible to study political and foreign-policy positions using both textual and behavioral evidence.

## Installation

Clone the repository and install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The manifesto analysis uses the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Reproducibility

The repository contains the code, notebooks and intermediate data required to reproduce the analyses, subject to the availability and redistribution conditions of the original data sources.

The analyses are primarily developed in Python and Jupyter notebooks.
