import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def fill_rile_by_party_country(data, method="mean"):
    data = data.copy()

    if method not in ["mean", "median"]:
        raise ValueError("method must be either 'mean' or 'median'")

    data["rile_original"] = data["rile"]

    missing_before = data["rile"].isna().sum()

    group_value = (
        data
        .groupby(["countryname", "partyname"])["rile"]
        .transform(method)
    )

    data["rile_filled"] = data["rile"].fillna(group_value)

    data["rile_was_imputed"] = (
        data["rile_original"].isna() & data["rile_filled"].notna()
    )

    missing_after_imputation = data["rile_filled"].isna().sum()

    data = data.dropna(subset=["rile_filled"]).copy()
    data["rile"] = data["rile_filled"]

    print("RILE missing-value treatment")
    print("----------------------------")
    print(f"Method: party-country {method}")
    print(f"Missing RILE values before imputation: {missing_before}")
    print(f"Values imputed: {data['rile_was_imputed'].sum()}")
    print(f"Rows dropped: {missing_after_imputation}")
    print(f"Missing RILE values after cleaning: {data['rile'].isna().sum()}")

    return data

def plot_country_parties(df, country, parties=None):
    """
    Show RILE evolution over time for a given party in a given country.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns
        ['countryname', 'partyname', 'date', 'rile']
    country : str
        The name of the country for which to display data
    parties : list of str, optional
        A list of party names to display. If None, all parties are displayed.

    """

    try:
        data = df[df['countryname'] == country].copy()
    except KeyError:
        print("Aucune donnée trouvée.")
        return

    if parties is not None:
        data = data[data['partyname'].isin(parties)]

    if data.empty:
        print("Aucune donnée trouvée.")
        return

    plt.figure(figsize=(12, 8))

    for party, group in data.groupby("partyname"):
        group = group.sort_values("date")
        plt.plot(
            group["date"],
            group["rile"],
            marker="o",
            label=party
        )

    plt.title(f"Evolution of the RILE index for parties in {country}")
    plt.xlabel("Year")
    plt.ylabel("RILE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Return RILE values each year for each party
    return data[["date", "partyname", "rile"]]


def run_wordscores_usa(
    manifesto_dir,
    a_priori_scores,
    reference_years_list,
    virgin_years_list,
    parties=("Democratic Party", "Republican Party"),
    party_file_prefixes=("dem", "rep"),
    token_pattern=r"(?u)\b\w+\b",
    verbose=True
):
    """
    Estimate Wordscores for virgin texts using reference texts.

    Parameters
    ----------
    manifesto_dir : str
        Path to the manifesto CSV folder, e.g. "../manifestos/usa".
    a_priori_scores : array-like
        Reference text scores, ordered in the same order as the reference texts:
        Democratic years first, then Republican years.
    reference_years_list : list
        Years used as reference texts.
    virgin_years_list : list
        Years used as virgin texts.
    parties : tuple
        Party names used for indexing.
    party_file_prefixes : tuple
        File prefixes corresponding to each party, e.g. ("dem", "rep").
    token_pattern : str
        Token pattern used by CountVectorizer.
    verbose : bool
        Whether to print intermediate summaries.

    Returns
    -------
    dict
        Dictionary containing reference texts, virgin texts, frequency matrix,
        word scores, raw virgin scores, and rescaled virgin scores.

    """

    ######################### Loadings of data ######################### 

    def load_manifesto_text(prefix, year):
        df = pd.read_csv(f"{manifesto_dir}/{prefix}_{year}.csv")
        return " ".join(df.iloc[:, 0].dropna().astype(str)) # Join all non-empty cells in the first column into a single string. Due to the structure of the CSV files, 
        # the manifesto text is in the first column, and we need to concatenate all rows to get the full text of the manifesto for that year.

    # Load reference texts
    reference_texts = []
    reference_labels = []

    for party, prefix in zip(parties, party_file_prefixes):
        for year in reference_years_list:
            reference_texts.append(load_manifesto_text(prefix, year))
            reference_labels.append((party, year))

    # Load virgin texts
    virgin_texts = []
    virgin_labels = []

    for party, prefix in zip(parties, party_file_prefixes):
        for year in virgin_years_list:
            virgin_texts.append(load_manifesto_text(prefix, year))
            virgin_labels.append((party, year))



    ##########################  Create the reference word-frequency matrix ######################### 

    vectorizer = CountVectorizer(token_pattern=token_pattern)
    ref_matrix = vectorizer.fit_transform(reference_texts).toarray()
    words = vectorizer.get_feature_names_out()

    reference_index = pd.MultiIndex.from_tuples(
        reference_labels,
        names=["partyname", "date"]
    )

    reference_freq = pd.DataFrame(
        ref_matrix,
        index=reference_index,
        columns=words
    )

    word_totals_per_text = reference_freq.sum(axis=1)
    reference_freq = reference_freq.div(word_totals_per_text.replace(0, np.nan), axis=0)

    # Compute conditional probabilities P(Text | Word)
    freq_totals = reference_freq.sum(axis=0)
    p_text_given_word = reference_freq.div(freq_totals.replace(0, np.nan), axis=1)

    # Compute word scores
    a_priori_scores = np.asarray(a_priori_scores)

    if len(a_priori_scores) != len(reference_texts):
        raise ValueError(
            "a_priori_scores must have the same length as reference_texts. "
            f"Got {len(a_priori_scores)} scores and {len(reference_texts)} texts."
        )

    word_scores = p_text_given_word.T.dot(a_priori_scores)

    df_words = pd.DataFrame(
        {"Word Score": word_scores},
        index=words
    )

    
    ######################### Virgin texts #########################
    # Transform virgin texts using the reference vocabulary
    virgin_matrix = vectorizer.transform(virgin_texts).toarray()

    # Compute relative word frequencies in each virgin text
    virgin_totals = virgin_matrix.sum(axis=1, keepdims=True)

    p_word_given_virgin = np.divide(
        virgin_matrix,
        virgin_totals,
        out=np.zeros_like(virgin_matrix, dtype=float),
        where=virgin_totals != 0
    )

    ######################### Compute raw virgin scores a,nd variances ######################### 

    raw_virgin_scores = p_word_given_virgin.dot(word_scores.to_numpy())

    virgin_index = pd.MultiIndex.from_tuples(
        virgin_labels,
        names=["partyname", "date"]
    )

    virgin_scores = pd.DataFrame(
        {"Raw Score": raw_virgin_scores},
        index=virgin_index
    )

    # Compute the variance of a virgin text

    v_scores = list(virgin_scores["Raw Score"])
    modified_word_scores = []

    for score in v_scores:
        modified_word_scores.append(list((word_scores-score)**2))

    modified_word_scores = np.array(modified_word_scores)
    raw_variances = np.sum(p_word_given_virgin*modified_word_scores, axis=1)
    raw_standard_errors = np.sqrt(raw_variances)/np.sqrt(np.shape(p_word_given_virgin)[1])
    virgin_scores["Standard Error"] = raw_standard_errors

    # Computing 95% confidence intervals for the virgin scores
    virgin_scores["Raw Lower CI"] = virgin_scores["Raw Score"] - 2 * virgin_scores["Standard Error"]
    virgin_scores["Raw Upper CI"] = virgin_scores["Raw Score"] + 2 * virgin_scores["Standard Error"]

    ##########################  Rescale virgin scores onto the reference-score scale ######################### 

    reference_sd = np.std(a_priori_scores, ddof=1)

    virgin_mean = np.mean(raw_virgin_scores)
    virgin_sd = np.std(raw_virgin_scores, ddof=1)

    if len(raw_virgin_scores) > 1 and virgin_sd != 0:
        rescaled_virgin_scores = (
            ((raw_virgin_scores - virgin_mean) / virgin_sd)
            * reference_sd
            + virgin_mean
        )
        
        rescaled_CI_lower = (
            ((virgin_scores["Raw Lower CI"] - virgin_mean) / virgin_sd)
            * reference_sd
            + virgin_mean
        )
        rescaled_CI_upper = (
            ((virgin_scores["Raw Upper CI"] - virgin_mean) / virgin_sd)
            * reference_sd
            + virgin_mean
        )
    else:
        rescaled_virgin_scores = np.full_like(raw_virgin_scores, np.nan, dtype=float)
        rescaled_CI_lower = np.full_like(raw_virgin_scores, np.nan, dtype=float)
        rescaled_CI_upper = np.full_like(raw_virgin_scores, np.nan, dtype=float)

    virgin_scores["Score"] = rescaled_virgin_scores
    virgin_scores["Lower CI"] = rescaled_CI_lower
    virgin_scores["Upper CI"] = rescaled_CI_upper


    if verbose:
        print("--- Frequency matrix of the reference texts ---")
        print(reference_freq)

        print("\n--- Computed scores for each word ---")
        print(df_words.round(2))

        print("\n--- Virgin text scores ---")
        print(virgin_scores.round(2))

    
    return {
        "reference_texts": reference_texts,
        "virgin_texts": virgin_texts,
        "reference_freq": reference_freq,
        "word_scores": df_words,
        "virgin_scores": virgin_scores,
        "vectorizer": vectorizer,
    }


def plot_true_vs_predicted_rile(virgin_scores, rile_values, score_col="Score", display_ci=True, party_names=None):
    """
    Plot true RILE values against predicted Wordscores values for each party.

    Parameters
    ----------
    virgin_scores : pd.DataFrame
        DataFrame indexed by partyname and date, with predicted scores.
    rile_values : pd.DataFrame
        DataFrame containing true RILE values with columns partyname, date, rile.
    score_col : str
        Column in virgin_scores containing predicted values.
    display_ci : bool
        Whether to display confidence intervals.
    party_names : list of str, optional
        List of parties to include in the plot. If None, all parties are included.  
    """

    predicted = (
        virgin_scores[[score_col,"Lower CI", "Upper CI"]]
        .rename(columns={score_col: "predicted_score"})
        .reset_index()
    )

    true_values = rile_values[["partyname", "date", "rile"]].rename(
        columns={"rile": "true_rile"}
    )

    plot_data = predicted.merge(
        true_values,
        on=["partyname", "date"],
        how="inner"
    )

    plot_data = plot_data[plot_data["partyname"].isin(party_names)] if party_names is not None else plot_data

    #   RMSE

    #plot_data["deviation"] = plot_data["true_rile"]-plot_data["predicted_score"]
    
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10.colors

    for i, (party, group) in enumerate(plot_data.groupby("partyname")):

        group = group.sort_values("date")
        if party_names is not None and len(party_names) <= 1:
            color = "red"
        else:
            color = colors[i % len(colors)]
        
        plt.plot(
            group["date"],
            group["true_rile"],
            marker="o",
            linestyle="-",
            label=f"{party} - true"
        )

        plt.plot(
            group["date"],
            group["predicted_score"],
            marker="x",
            linestyle="--",
            label=f"{party} - predicted"
        )

        if display_ci:
            plt.errorbar(
                group["date"].to_numpy(),
                group["predicted_score"].to_numpy(),
                yerr=np.vstack([
                    (group["predicted_score"] - group["Lower CI"]).to_numpy(),
                    (group["Upper CI"] - group["predicted_score"]).to_numpy()
                ]),
                fmt="none",
                capsize=4,
                alpha=0.7,
                ecolor=color
            )

    plt.title("True vs predicted RILE values")
    plt.xlabel("Year")
    plt.ylabel("RILE")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_iterative_wordscores(
    years_list,
    rile_values,
    manifesto_dir="../manifestos/usa",
    parties=("Democratic Party", "Republican Party"),
    min_reference_years=1,
    verbose=False
):
    """
    Run Wordscores iteratively with an expanding set of reference years.

    At each step:
    - reference years are all years up to the current cutoff year
    - virgin years are all years after the cutoff year
    """

    all_results = {}

    for i in range(min_reference_years, len(years_list)):
        reference_years_list = years_list[:i]
        virgin_years_list = years_list[i:]

        useful_rile_values = rile_values[
            rile_values["partyname"].isin(parties)
            & rile_values["date"].isin(reference_years_list)
        ].copy()

        useful_rile_values["partyname"] = pd.Categorical(
            useful_rile_values["partyname"],
            categories=parties,
            ordered=True
        )

        useful_rile_values = useful_rile_values.sort_values(
            ["partyname", "date"]
        )

        a_priori_scores = useful_rile_values["rile"].to_numpy()

        if verbose:
            print("\n" + "=" * 60)
            print(f"Iteration {i}")
            print(f"Reference years: {reference_years_list}")
            print(f"Virgin years: {virgin_years_list}")
            print("=" * 60)

        results = run_wordscores_usa(
            manifesto_dir=manifesto_dir,
            a_priori_scores=a_priori_scores,
            reference_years_list=reference_years_list,
            virgin_years_list=virgin_years_list,
            parties=parties,
            party_file_prefixes=("dem", "rep"),
            verbose=False
        )

        all_results[i] = results["virgin_scores"]

    return all_results