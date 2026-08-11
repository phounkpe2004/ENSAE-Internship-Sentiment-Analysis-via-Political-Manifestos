import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def compute_sentiment(section_dataframe):
    
    def score_computation_by_row(party_text):
        analyzer = SentimentIntensityAnalyzer()

        scores_list = []
        for sentence in party_text.split("."):
            vs = analyzer.polarity_scores(sentence)
            scores_list.append(vs['compound']) # compound score of vaderSentiment

        return np.mean(scores_list)
        
    section_dataframe["sentiment"] = section_dataframe["text"].apply(score_computation_by_row)

    return section_dataframe

def plot_sentiment(section_dataframe):

    plt.figure(figsize=(12, 6))

    for party in section_dataframe["party"].unique():
        plt.plot(
            section_dataframe[section_dataframe["party"]==party]["year"],
            section_dataframe[section_dataframe["party"]==party]["sentiment"],
            marker="o",
            linewidth=2,
            label=party
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    plt.title("Global Sentiment of Parties on Foreign Affairs")
    plt.xlabel("Year")
    plt.ylabel("Sentiment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def setup_sentiment_analyzer(nlp, country_synonyms_dict):
    """
    Initialise et configure le pipeline SpaCy avec l'EntityRuler pour les pays,
    et instancie l'analyseur de sentiment VADER.
    
    Retourne :
        nlp : Le pipeline SpaCy configuré.
        analyzer : L'instance de SentimentIntensityAnalyzer.
    """
    
    analyzer = SentimentIntensityAnalyzer()

    # Ajout du composant EntityRuler avant le NER standard
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    # Génération et ajout des patterns dynamiques
    patterns = []
    for country_label, alias_list in country_synonyms_dict.items():
        for alias in alias_list:
            patterns.append({"label": country_label, "pattern": alias})
            
    ruler.add_patterns(patterns)
    
    return analyzer

def analyze_targeted_sentiment(text_segment, nlp, analyzer, country_synonyms_dict):
    """
    Analyzes VADER sentiment sentence-by-sentence on a pre-parsed spaCy Doc.
    """
    doc = nlp(text_segment)
    sentiment_accumulator = {}
    
    for sentence in doc.sents:
        # Detect countries present in this sentence
        detected_countries = {
            ent.label_ for ent in sentence.ents 
            if ent.label_ in country_synonyms_dict
        }
        
        if detected_countries:
            vader_score = analyzer.polarity_scores(sentence.text)['compound']
            
            for country in detected_countries:
                sentiment_accumulator.setdefault(country, []).append(vader_score)
                
    # Calculate average sentiment scores and mention counts
    final_scores = {}
    number_mentions = {}

    for country, scores in sentiment_accumulator.items():
        final_scores[country] = sum(scores) / len(scores)
        number_mentions[country] = len(scores)
        
    sorted_scores = dict(sorted(final_scores.items(), key=lambda x: x[0]))
    sorted_mentions = dict(sorted(number_mentions.items(), key=lambda x: x[0]))

    return sorted_scores, sorted_mentions


def compute_country_scores(section_dataframe, nlp, analyzer, country_synonyms_dict, batch_size=50):
    """
    Applies targeted sentiment analysis using SpaCy's high-performance nlp.pipe batching.
    """
    df = section_dataframe.copy()
    
    texts = df["text"].fillna("").astype(str).tolist()
    docs = nlp.pipe(texts, batch_size=batch_size)
    
    # Extract results in a single pass
    sentiments = []
    mentions = []
    
    for doc in docs:
        score_dict, mention_dict = analyze_targeted_sentiment(
            text_segment=doc,
            nlp=nlp,
            analyzer=analyzer, 
            country_synonyms_dict=country_synonyms_dict
        )
        sentiments.append(score_dict)
        mentions.append(mention_dict)
        
    # Assign directly to the DataFrame
    df["sentiment_by_countries"] = sentiments
    df["number_mentions"] = mentions

    return df


def plot_country_sentiment(dataframe, party, year, figsize=(10, 12)):

    filtered_row = dataframe[(dataframe["party"] == party) & (dataframe["year"] == year)]
    
    if filtered_row.empty:
        print(f"[Warning] No data found for Party: '{party}' and Year: {year}")
        return

    country_sentiment = filtered_row.iloc[0]["sentiment_by_countries"]

    if not isinstance(country_sentiment, dict) or not country_sentiment:
        print(f"[Warning] No country sentiment data available for {party} in {year}.")
        return


    plot_df = pd.DataFrame(list(country_sentiment.items()), columns=["Country", "Sentiment"])
    #plot_df = plot_df[plot_df['Sentiment'] != 0.0]
    
    if plot_df.empty:
        print(f"[Warning] All sentiment scores were 0.0 for {party} in {year}. Nothing to plot.")
        return

    plot_df = plot_df.sort_values(by="Sentiment", ascending=True)
    plot_df["Country_Clean"] = plot_df["Country"].str.replace("_", " ").str.title()

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#e53e3e" if x < 0 else "#3182ce" for x in plot_df["Sentiment"]]

    ax.hlines(
        y=plot_df["Country_Clean"],
        xmin=0,
        xmax=plot_df["Sentiment"],
        color=colors,
        alpha=0.7,
        linewidth=2,
    )
    ax.scatter(
        plot_df["Sentiment"],
        plot_df["Country_Clean"],
        color=colors,
        s=70,
        alpha=1.0,
        edgecolors="black",
        zorder=3,
    )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    ax.set_xlim(-1.0, 1.0)
    ax.set_title(
        f"Net Foreign Policy Sentiment by Country\n({party} — {year})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Net Sentiment Score (-1.0 to +1.0)", fontsize=11)
    ax.set_ylabel("")

    sns.despine(left=True, bottom=False)

    plt.tight_layout()
    
    plt.show()

def get_country_sentiment(sentiment_dataset, country="AUSTRALIA"):
    expanded = pd.DataFrame(
        sentiment_dataset["sentiment_by_countries"].tolist(), 
        index=sentiment_dataset["year"]
    )
    
    if country in expanded.columns:
        # 2. Extract country column & aggregate duplicate years (e.g., take mean per year)
        # Leaving missing years as NaN so forward filling actually works
        scores = expanded[country].groupby(level=0).mean()
    else:
        # If the country isn't present in any dictionary
        scores = pd.Series(dtype=float)

    return scores.to_dict()

def get_country_mentions(sentiment_dataset, country="AUSTRALIA"):
    expanded = pd.DataFrame(
        sentiment_dataset["number_mentions"].tolist(), 
        index=sentiment_dataset["year"]
    )
    
    if country in expanded.columns:
        # 2. Extract country column & aggregate duplicate years (e.g., take mean per year)
        # Leaving missing years as NaN so forward filling actually works
        scores = expanded[country].groupby(level=0).sum()
    else:
        # If the country isn't present in any dictionary
        scores = pd.Series(dtype=float)

    return scores.to_dict()