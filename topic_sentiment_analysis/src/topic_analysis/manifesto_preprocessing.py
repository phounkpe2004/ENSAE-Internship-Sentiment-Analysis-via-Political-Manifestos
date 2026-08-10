import os
from os.path import isfile, join
from typing import Optional, TypedDict

import pandas as pd
import parse
from spacy import Language


class RawManifesto(TypedDict):
	party: str
	year: int
	raw_text: str
	
def structure_manifestos(manifesto_dir, party_prefix_mapper, covered_years: Optional[int] = None, name_format="{prefix}_{year}.csv"):
	"""
    Load manifesto files and aggregate them into a single DataFrame.

    Each manifesto is read from a CSV file, the first column is concatenated
    into a single text string, and metadata (party and year) are extracted
    from either the provided years or the filenames.

    Args:
        manifesto_dir (str): Directory containing the manifesto CSV files.
        party_prefix_mapper (dict): Mapping from filename prefixes to party names.
        covered_years (Optional[Iterable[int]]): Years to load for every party.
            If None, all files in ``manifesto_dir`` matching ``name_format`` are
            processed.
        name_format (str): Filename pattern used to locate or parse manifesto
            files. Must contain ``{prefix}`` and ``{year}`` placeholders.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - ``party``: Party name.
            - ``year``: Election year.
            - ``raw_text``: Concatenated manifesto text.
    """
		
	dico = {"party":[], "year": [],"raw_text": []}

	if covered_years:
		for prefix,party in party_prefix_mapper.items():
			for year in covered_years:
				df = pd.read_csv(f"{manifesto_dir}/"+name_format.format(prefix=prefix,year=year))
				text = " ".join(df.iloc[:, 0].dropna().astype(str))

				dico["party"].append(party)
				dico["year"].append(year)
				dico["raw_text"].append(text)
	else:
		file_list = [x for x in os.listdir(manifesto_dir) if isfile(join(manifesto_dir,x))]
		for file in file_list:
			df = pd.read_csv(join(manifesto_dir,file))
			result_dict = parse.parse(name_format, file).named
			prefix, year = result_dict["prefix"], result_dict["year"]

			text = " ".join(df.iloc[:, 0].dropna().astype(str))

			dico["party"].append(party_prefix_mapper[prefix])
			dico["year"].append(year)
			dico["raw_text"].append(text)


	return pd.DataFrame(dico)


class ProcessedManifesto(TypedDict):
	chunk_id: int
	sentence_ids: list[int]
	original_sentences: str
	clean_sentences: list[str]
	chunk_text: str

def preprocess_manifesto(text: str, nlp: Language, sentences_per_chunk=1):

	"""
    Preprocess manifesto text into cleaned sentence chunks.

    The text is segmented into sentences, tokenized with spaCy, and cleaned by
    lemmatizing tokens, lowercasing, and removing stop words, punctuation, and
    whitespace. Consecutive sentences are then grouped into fixed-size chunks.

    Args:
        text (str): Raw manifesto text.
        nlp (Language): spaCy language pipeline used for processing.
        sentences_per_chunk (int): Number of consecutive sentences to include in each chunk.

    Returns:
        list[dict]: A list of chunk records containing chunk IDs, sentence IDs,
        original sentences, cleaned sentences, and concatenated cleaned text.
    """

	doc = nlp(str(text))

	sentence_records = []

	for sent_id, sent in enumerate(doc.sents):
		clean_words = [
			token.lemma_.lower()
			for token in sent
			if not token.is_stop
			and not token.is_punct
			and not token.is_space
		]

		if clean_words:
			sentence_records.append({
				"sent_id": sent_id,
				"original_sentence": sent.text,
				"clean_sentence": " ".join(clean_words)
			})

	chunk_records = []

	for chunk_id, start in enumerate(range(0, len(sentence_records), sentences_per_chunk)):
		selected_sentences = sentence_records[start:start + sentences_per_chunk]

		chunk_records.append({
			"chunk_id": chunk_id,
			"sentence_ids": [s["sent_id"] for s in selected_sentences],
			"original_sentences": " ".join(s["original_sentence"] for s in selected_sentences),
			"clean_sentences": [s["clean_sentence"] for s in selected_sentences],
			"chunk_text": " ".join(s["clean_sentence"] for s in selected_sentences)
		})

	return chunk_records
