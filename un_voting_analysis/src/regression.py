import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def perform_regression(data, x_cols, y_col, method='HC3', plotnum=0):
    """
    Effectue une régression linéaire multiple.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant les variables explicatives et la variable expliquée.

    x_cols : list[str]
        Colonnes explicatives.

    y_col : str
        Colonne expliquée.

    method : str
        Type de covariance pour erreurs robustes ('HC3' par défaut).

    Returns
    -------
    model : RegressionResults
        L'objet modèle ajusté de statsmodels.
    """

    cols = x_cols + [y_col]
    df = data[cols].dropna().copy()

    X = sm.add_constant(df[x_cols])
    y = df[y_col]

    model = sm.OLS(y, X).fit(
        method='qr',
        cov_type=method
    )

    # Regression summary
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0, 0.5,
        model.summary().as_text(),
        va='top',
        ha='center',
        fontsize=10,
        family='monospace'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Scatter plot + regression line
    plt.figure(figsize=(7, 5))

    sns.regplot(
        data=df,
        x=x_cols[plotnum],
        y=y_col,
        ci=95
    )

    plt.xlabel(x_cols[plotnum])
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

    return model


def perform_sentiment_regressions(data, country="AUS", method="HC3"):

    df = data[data["country"] == country].copy()
    df = df.sort_values("year")

    # Model A: distance_t ~ sentiment_t
    model_a = perform_regression(
        df,
        x_cols=["country_sentiment_towards_usa"],
        y_col="distance_to_usa",
        method=method
    )

    # Lagged sentiment
    df["sentiment_lag1"] = (
        df["country_sentiment_towards_usa"].shift(1)
    )

    df["sentiment_lag2"] = (
        df["country_sentiment_towards_usa"].shift(2)
    )

    # Model B: distance_t ~ sentiment_(t-1)
    model_b = perform_regression(
        df,
        x_cols=["sentiment_lag1"],
        y_col="distance_to_usa",
        method=method
    )


    # Model C: distance_t ~ sentiment_(t-1), sentiment_(t-2)
    model_c = perform_regression(
        df,
        x_cols=["sentiment_lag1", "sentiment_lag2"],
        y_col="distance_to_usa",
        method=method
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c
    }


def granger_causality_test(data, country="AUS", maxlag=2):
    
    df = (
        data[data["country"] == country]
        [["year", "distance_to_usa", "country_sentiment_towards_usa"]]
        .sort_values("year")
        .dropna()
        .copy()
    )

    print(f"Country: {country}")
    print(f"Observations: {len(df)}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}\n")

    result_s_to_d = grangercausalitytests(
        df[["distance_to_usa", "country_sentiment_towards_usa"]],
        maxlag=maxlag,
        verbose=False
    )


    result_d_to_s = grangercausalitytests(
        df[["country_sentiment_towards_usa", "distance_to_usa"]],
        maxlag=maxlag,
        verbose=False
    )

    return {
        "sentiment_to_distance": result_s_to_d,
        "distance_to_sentiment": result_d_to_s
    }

def extract_granger_pvalues(results):
    
    rows = []

    for direction, result in results.items():
        for lag, tests in result.items():
            f_test = tests[0]["ssr_ftest"]
            
            rows.append({
                "direction": direction,
                "lag": lag,
                "F-statistic": f_test[0],
                "p-value": f_test[1]
            })

    return pd.DataFrame(rows)