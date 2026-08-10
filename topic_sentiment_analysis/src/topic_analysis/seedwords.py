import tomotopy as tmt
from topic_analysis import manifesto_preprocessing as mp
import pandas as pd
from spacy import Language
from typing import Optional

def generate_seed_words(manifesto_dataframe: pd.DataFrame, nlp: Language, seed: Optional[int] = None, k=8, top_n_words=100):
    """
    Train an LDA model on manifesto text and extract the top words for each topic.

    Manifestos are preprocessed into fixed-size sentence chunks, converted into
    token lists, and used to train a tomotopy LDA model. The function returns
    the most representative words for each learned topic.

    Args:
        manifesto_dataframe (pd.DataFrame): DataFrame containing a ``raw_text``
            column with manifesto text.
        nlp (Language): spaCy language pipeline used for preprocessing.
        seed (int): Random seed for reproducible LDA training.
        k (int): Number of topics to infer.
        top_n_words (int): Number of top words to return per topic.

    Returns:
        list[str]: Top-ranked words identified. Now you need to map them to your categories either by hand or using an LLM.
    """

    tmt_model = tmt.LDAModel(k=k,seed=seed,min_cf=3,rm_top=0)

    data = manifesto_dataframe

    # Add documents to simple LDA

    for _, row in data.iterrows():
        chunks = mp.preprocess_manifesto(
            text=row["raw_text"],
            nlp=nlp,
            sentences_per_chunk=5
        )

        for chunk in chunks:
            tokens = chunk["chunk_text"].split()

            if len(tokens) > 3:
                tmt_model.add_doc(words=tokens)


    # Train model

    tmt_model.burn_in = 100
    tmt_model.train(0)

    for _ in range(10):
        tmt_model.train(100)


    # Extract top topic words

    topic_words = {}

    for topic_id in range(tmt_model.k):
        topic_words[topic_id] = [
            word
            for word, _ in tmt_model.get_topic_words(topic_id, top_n=top_n_words)
        ]

    return [x for xs in topic_words.values() for x in xs]

# unseeded_topics = run_lda_pipeline_for_seed(seed=42, k=8, top_n_words=50)["topic_words"] # MUST BE EXECUTED TO COMPUTE A LDA BASED DICT