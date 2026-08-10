import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt


NLP = spacy.load(
    "en_core_web_sm",
    disable=["tagger", "attribute_ruler", "lemmatizer", "ner"],
)

def structure_manifestos(manifesto_dir, party_prefix_mapper, covered_years):

    dico = {"party":[], "year": [],"raw_text": []}

    for prefix,party in party_prefix_mapper.items():
        for year in covered_years:
            manifestos_dataframe = pd.read_csv(f"{manifesto_dir}/{prefix}_{year}.csv")
            text = " ".join(manifestos_dataframe.iloc[:, 0].dropna().astype(str)) # Similar to the file Wordscore

            dico["party"].append(party)
            dico["year"].append(year)
            dico["raw_text"].append(text)

    return pd.DataFrame(dico)

def extract_distribution_from_doc(doc):
    words_per_sentence = [
        sum(
            1
            for token in sent
            if not token.is_stop and not token.is_punct and not token.is_space if len(token) > 2
        )
        for sent in doc.sents
    ]

    number_sentences = len(words_per_sentence)
    total_words = sum(words_per_sentence)
    word_per_sentences = np.ceil(total_words / number_sentences) if number_sentences else 0

    return word_per_sentences, number_sentences, total_words


def add_manifesto_distributions(manifestos_dataframe, nlp=NLP, text_col="raw_text", batch_size=8):
    docs = nlp.pipe(
        manifestos_dataframe[text_col].fillna("").astype(str),
        batch_size=batch_size,
    )

    stats = [extract_distribution_from_doc(doc) for doc in docs]

    stats_manifestos_dataframe = pd.DataFrame(
        stats,
        columns=["word_per_sentences", "number_sentences", "total_words"],
        index=manifestos_dataframe.index,
    )

    return manifestos_dataframe.join(stats_manifestos_dataframe)


def plot_manifesto_stats(manifestos_df, party=None):
    manifestos_dataframe = manifestos_df.copy()

    if party is not None:
        manifestos_dataframe = manifestos_dataframe[manifestos_dataframe["party"] == party].copy()

    manifestos_dataframe = manifestos_dataframe.sort_values("year")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    x = np.arange(len(manifestos_dataframe))
    labels = manifestos_dataframe["year"].astype(str)

    ax1.bar(
        x,
        manifestos_dataframe["word_per_sentences"],
        alpha=0.6,
        label="Average words per sentence",
        color="tab:blue",
    )
    ax1.set_ylabel("Words per sentence", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        manifestos_dataframe["number_sentences"],
        marker="o",
        alpha=0.6,
        label="Number of sentences",
        color="tab:orange",
    )
    ax2.set_ylabel("Number of sentences", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(
        x,
        manifestos_dataframe["total_words"],
        marker="s",
        alpha=0.6,
        label="Total words",
        color="tab:green",
    )
    ax3.set_ylabel("Total words", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel("Year")

    title = "Manifesto Text Statistics"
    if party is not None:
        title += f" - {party}"
    ax1.set_title(title)

    handles_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
    handles, labels = [sum(item, []) for item in zip(*handles_labels)]
    ax1.legend(handles, labels, loc="upper left")

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


from collections import Counter

def lexical_diversity_metrics(text, nlp=NLP):
    doc = nlp(str(text))

    tokens = [
        token.text.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
    ]

    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    if total_tokens == 0:
        return {
            "num_tokens": 0,
            "num_types": 0,
            "ttr": 0,
            "root_ttr": 0,
            "guiraud": 0,
            "hapax_count": 0,
            "hapax_rate": 0,
        }

    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)

    return {
        "num_tokens": total_tokens,
        "num_types": unique_tokens,
        "ttr": unique_tokens / total_tokens,
        "guiraud": unique_tokens / np.sqrt(total_tokens),
        "hapax_count": hapax_count,
        "hapax_rate": hapax_count / unique_tokens,
    }


def add_lexical_diversity_metrics(structured_manifestos, nlp=NLP, text_col="raw_text"):
    metrics = structured_manifestos[text_col].apply(
        lambda text: lexical_diversity_metrics(text, nlp)
    )

    metrics_manifestos_dataframe = pd.DataFrame(metrics.tolist(), index=structured_manifestos.index)

    return structured_manifestos.join(metrics_manifestos_dataframe)

def plot_manifesto_metrics(
    manifestos_dataframe,
    metrics,
    party=None,
    alpha=0.6,
    figsize=(12, 6),
):

    if party is not None:
        manifestos_dataframe = manifestos_dataframe[manifestos_dataframe["party"] == party].copy()

    manifestos_dataframe = manifestos_dataframe.sort_values("year")

    fig, ax = plt.subplots(figsize=figsize)

    for metric in metrics:
        if metric == "hapax_rate":
            ax.plot(
                manifestos_dataframe["year"],
                manifestos_dataframe[metric]*100,
                marker="o",
                alpha=alpha,
                label=metric,
            )
        else:
            ax.plot(
                manifestos_dataframe["year"],
                manifestos_dataframe[metric],
                marker="o",
                alpha=alpha,
                label=metric,
            )

    title = "Linguistic Metrics"
    if party is not None:
        title += f" - {party}"

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()