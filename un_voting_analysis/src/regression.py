import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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