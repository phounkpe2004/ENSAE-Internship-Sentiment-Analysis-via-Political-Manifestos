# Data

This directory contains the datasets used throughout the project.

## Organization

The data are organized into three categories according to their role in the analysis:

```text
data/
├── raw/
├── intermediate/
└── processed/
```

### `raw/`

Contains the original manifesto texts used as input data.

The manifesto files are organized by country:

```text
raw/
└── manifestos/
    ├── australia/
    └── usa/
```

The files in this directory should be treated as source data and should not be modified directly.

### `intermediate/`

Contains datasets generated during intermediate stages of the analysis.

For example, the `foreign_affairs/` directory contains text datasets extracted or prepared for the foreign-affairs-related analyses.

```text
intermediate/
└── foreign_affairs/
```

### `processed/`

Contains structured datasets prepared for analysis.

This directory currently includes `manifesto_data.csv`, which provides structured manifesto-level data used by the project.

## Data workflow

The general data workflow is:

```text
Raw manifesto texts
        │
        ▼
   Preprocessing
        │
        ▼
Intermediate data (Could be also directly used for analysis)
        │
        ▼
 Processed data (eventually)
        │
        ▼
     Analysis
```

The distinction between these directories is intended to preserve the provenance of the data and make the analysis workflow easier to follow.
