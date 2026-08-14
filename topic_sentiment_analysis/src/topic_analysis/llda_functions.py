import tomotopy as tmt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from topic_analysis import manifesto_preprocessing as mp

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
) # Stylish graphics

def run_llda_pipeline_for_seed(manifestos_dataframe, nlp , categories, seed, seed_words_dict, used_parties=None):

	tmt_model = tmt.LLDAModel(seed=seed)

	doc_metadata = []

	if used_parties is None:     

		for _, row in manifestos_dataframe.iterrows():
			chunks = mp.preprocess_manifesto(
				text=row["raw_text"],
				nlp=nlp,
				sentences_per_chunk=5
			)

			for chunk in chunks:
				tokens = chunk["chunk_text"].split()

				if len(tokens) > 3:
					assigned_labels = []

					for category, keywords in seed_words_dict.items():
						if any(keyword in tokens for keyword in keywords):
							assigned_labels.append(category)

					if not assigned_labels:
						assigned_labels = categories

					tmt_model.add_doc(
						words=tokens,
						labels=assigned_labels
					)

					doc_metadata.append({
						"manifesto_id": row.get("manifesto_id", None),
						"party": row["party"],
						"year": row["year"],
						"chunk_id": chunk["chunk_id"],
						"chunk": chunk["chunk_text"],
						"original_sentences": chunk["original_sentences"],
						"sentence_ids": chunk["sentence_ids"],
						"assigned_labels": assigned_labels
					})

	else:
		for _, row in manifestos_dataframe[manifestos_dataframe["party"].isin(used_parties)].iterrows():
			chunks = mp.preprocess_manifesto(
				text=row["raw_text"],
				nlp=nlp,
				sentences_per_chunk=5
			)

			for chunk in chunks:
				tokens = chunk["chunk_text"].split()

				if len(tokens) > 3:
					assigned_labels = []

					for category, keywords in seed_words_dict.items():
						if any(keyword in tokens for keyword in keywords):
							assigned_labels.append(category)

					if not assigned_labels:
						assigned_labels = categories

					tmt_model.add_doc(
						words=tokens,
						labels=assigned_labels
					)

					doc_metadata.append({
						"manifesto_id": row.get("manifesto_id", None),
						"party": row["party"],
						"year": row["year"],
						"chunk_id": chunk["chunk_id"],
						"chunk": chunk["chunk_text"],
						"original_sentences": chunk["original_sentences"],
						"sentence_ids": chunk["sentence_ids"],
						"assigned_labels": assigned_labels
					})


	tmt_model.burn_in = 100
	tmt_model.train(0)

	for _ in range(10):
		tmt_model.train(100)


	id_to_label = {
		topic_id: label
		for topic_id, label in enumerate(tmt_model.topic_label_dict)
	}

	label_to_id = {
		label: topic_id
		for topic_id, label in id_to_label.items()
	}


	topic_vectors = []

	for doc, meta in zip(tmt_model.docs, doc_metadata):
		topic_dist = doc.get_topic_dist()

		row_dict = {key: meta[key] for key in meta.keys()}

		for label, topic_id in label_to_id.items():
			row_dict[f"prob_{label}"] = topic_dist[topic_id]

		topic_vectors.append(row_dict)

	topic_df = pd.DataFrame(topic_vectors)

	prob_cols = [col for col in topic_df.columns if col.startswith("prob_")]

	topic_df["predicted_topic"] = (
		topic_df[prob_cols]
		.idxmax(axis=1)
		.str.replace("prob_", "", regex=False)
	)

	topic_df["seed"] = seed


	evaluation_report = evaluate_efficiency(
		tmt_model=tmt_model,
		doc_metadata=doc_metadata,
		categories=categories,
	)

	return {
		"seed": seed,
		"model": tmt_model,
		"topic_df": topic_df,
		"evaluation_report": evaluation_report
	}



def aggregate_evaluation_reports(seed_runs, lower_q=0.025, upper_q=0.975):

	drift_frames = []
	npmi_frames = []

	for run in seed_runs:

		seed = run["seed"]

		evaluation_report = run["evaluation_report"]

		drift_df = evaluation_report["temporal_drift_df"].copy()
		drift_df["seed"] = seed

		npmi_df = evaluation_report["npmi_df"].copy()
		npmi_df["seed"] = seed

		drift_frames.append(drift_df)
		npmi_frames.append(npmi_df)

	drift_df = pd.concat(drift_frames, ignore_index=True)
	npmi_df = pd.concat(npmi_frames, ignore_index=True)

	drift_summary = (
		drift_df
		.groupby(["label", "period"])["distance"]
		.agg(
			mean="mean",
			std="std",
			lower=lambda x: x.quantile(lower_q),
			upper=lambda x: x.quantile(upper_q)
		)
		.reset_index()
	)

	npmi_summary = (
		npmi_df
		.groupby("label")["npmi_coherence"]
		.agg(
			mean="mean",
			std="std",
			lower=lambda x: x.quantile(lower_q),
			upper=lambda x: x.quantile(upper_q)
		)
		.reset_index()
	)

	return drift_summary, npmi_summary


def aggregate_party_pair_hellinger_reports(
	seed_runs,
	party_a,
	party_b,
	topic=None,
	lower_q=0.025,
	upper_q=0.975
):

	rows = []

	for run in seed_runs:

		seed = run["seed"]

		topic_words_by_year_party = (
			run["evaluation_report"]["topic_words_by_year_party"]
		)

		df = compute_party_hellinger_distance(
			topic_words_by_year_party=topic_words_by_year_party,
			party_a=party_a,
			party_b=party_b,
			topic=topic
		)

		if df.empty:
			continue

		df["seed"] = seed

		rows.append(df)


	if not rows:
		return pd.DataFrame(), pd.DataFrame()


	drift_df = pd.concat(
		rows,
		ignore_index=True
	)


	summary_df = (
		drift_df
		.groupby(
			["label", "year", "party_a", "party_b"]
		)["distance"]
		.agg(
			mean="mean",
			std="std",
			lower=lambda x: x.quantile(lower_q),
			upper=lambda x: x.quantile(upper_q)
		)
		.reset_index()
	)


	return summary_df

def plot_drift_with_confidence_bounds(drift_summary):
	
	fig, ax = plt.subplots(figsize=(9, 5.5))

	for label in drift_summary["label"].unique():
		label_df = drift_summary[drift_summary["label"] == label].copy()

		ax.plot(
			label_df["period"],
			label_df["mean"],
			marker="o",
			linewidth=2,
			label=label
		)

		ax.fill_between(
			label_df["period"],
			label_df["lower"],
			label_df["upper"],
			alpha=0.2
		)

	ax.set_ylim(0, 1.0)

	ax.axhline(
		0.35,
		color="#38a169",
		linestyle=":",
		alpha=0.8
	)

	ax.set_title(
		"Year-over-Year Hellinger Drift",
		fontsize=12,
		fontweight="bold"
	)

	ax.set_xlabel("Year")
	ax.set_ylabel("Hellinger Distance")
	ax.legend(frameon=True)

	plt.tight_layout()
	plt.show()



def plot_npmi_with_confidence_bounds(npmi_summary):
	fig, ax = plt.subplots(figsize=(9, 5.5))

	labels = npmi_summary["label"]
	means = npmi_summary["mean"]

	lower_errors = means - npmi_summary["lower"]
	upper_errors = npmi_summary["upper"] - means

	errors = [lower_errors, upper_errors]

	colors = [
		"#3182ce" if score >= 0 else "#e53e3e"
		for score in means
	]

	ax.bar(
		labels,
		means,
		yerr=errors,
		capsize=5,
		color=colors,
		edgecolor="#4a5568",
		width=0.5
	)

	ax.axhline(0, color="black", linewidth=1.2)

	ax.set_title(
		"NPMI Topic Coherence",
		fontsize=12,
		fontweight="bold"
	)

	ax.set_ylabel("NPMI Coherence")

	plt.xticks(rotation=30, ha="right")
	plt.tight_layout()
	plt.show()


def plot_party_pair_hellinger_with_confidence(
	summary_df,
	topic=None,
	figsize=(10,5)
):

	df = summary_df.copy()

	if topic is not None:
		df = df[df["label"] == topic]


	plt.figure(figsize=figsize)


	for label in df["label"].unique():

		plot_df = (
			df[df["label"] == label]
			.sort_values("year")
		)

		plt.plot(
			plot_df["year"],
			plot_df["mean"],
			marker="o",
			linewidth=2,
			label=label
		)

		plt.fill_between(
			plot_df["year"],
			plot_df["lower"],
			plot_df["upper"],
			alpha=0.2
		)


	plt.ylim(0,1)

	plt.xlabel("Year")
	plt.ylabel("Hellinger Distance")

	title = (
		f"{df.iloc[0]['party_a']} vs "
		f"{df.iloc[0]['party_b']}: Hellinger Distance"
	)

	if topic:
		title += f" - {topic}"


	plt.title(title)

	plt.legend(title="Topic")
	plt.grid(alpha=0.3)

	plt.tight_layout()
	plt.show()


def evaluate_results(manifestos_dataframe, nlp , categories, seeds, seed_words_dict, parties):

	seed_runs = []
	party_label = parties if parties else "All Parties (Global)"
	print(f"\n=== Starting LLDA Pipeline for: {party_label} ===")
	
	for seed in seeds:
		print(f"  Running LLDA with seed {seed}...")
		result = run_llda_pipeline_for_seed(
			manifestos_dataframe=manifestos_dataframe, 
			nlp=nlp , 
			categories=categories,
			seed=seed,
			used_parties=parties,
			seed_words_dict=seed_words_dict
		)
		seed_runs.append(result)
		
	drift_summary, npmi_summary = aggregate_evaluation_reports(seed_runs)

	return drift_summary, npmi_summary

def comparison_results(manifestos_dataframe, nlp , categories, seeds, seed_words_dict, party_a, party_b):
	"""
	Exécute l'analyse LLDA pour une liste de partis, puis compare précisément 
	deux partis sélectionnés pour l'analyse Hellinger.
	"""
	combined_runs = []

	for seed in seeds:
		print(f"  Running LLDA with seed {seed}...")
		result = run_llda_pipeline_for_seed(
			manifestos_dataframe=manifestos_dataframe, 
			nlp=nlp , 
			categories=categories,
			seed=seed,
			used_parties=[party_a,party_b],
			seed_words_dict=seed_words_dict
		)
		combined_runs.append(result)
	
	# Hellinger drift between the two parties
	print(f"\n=== Generating Hellinger Analysis: {party_a} vs {party_b} ===")
	party_drift_summary = aggregate_party_pair_hellinger_reports(
		combined_runs,
		party_a=party_a,
		party_b=party_b
	)

	return party_drift_summary

def plot_comparison(results_dataframe, topic="Foreign_Affairs"):
	plot_party_pair_hellinger_with_confidence(results_dataframe,  topic=topic)

def extract_topic_sentences_by_party(manifestos_dataframe, nlp, categories, seed_words_dict, topic="Foreign_Affairs", seed=42):
    """
    Exécute le pipeline LLDA pour chaque parti unique du DataFrame, extrait les phrases 
    liées à un sujet spécifique (topic), et fusionne le texte par année.
    
    Retourne :
        Un DataFrame Pandas combiné contenant les colonnes ['year', 'text', 'party']
    """
    
    all_parties_records = []
    
    
    unique_parties = manifestos_dataframe["party"].unique()
    
    for party in unique_parties:
        print(f"Processing LLDA pipeline for: {party}...")
        
        # 1. 
        results = run_llda_pipeline_for_seed(
            manifestos_dataframe=manifestos_dataframe, 
            nlp=nlp, 
            categories=categories, 
            seed_words_dict=seed_words_dict, 
            seed=seed, 
            used_parties=[party] # Dynamique ici
        )
        
        topic_df = results["topic_df"]
        
        # 2.
        party_topic_sentences = (
            topic_df[topic_df["predicted_topic"] == topic]
            .sort_values(["year", "chunk_id"])
            .groupby("year")["original_sentences"]
            .apply(lambda x: " ".join(x.astype(str)))
            .reset_index(name="text")
        )
    
        party_topic_sentences["party"] = party
        
        # Ajout au catalogue global
        all_parties_records.append(party_topic_sentences)
        
    final_dataframe = pd.concat(all_parties_records, ignore_index=True)
    
    return final_dataframe


def build_sentiment_input_from_dimension(dimension_text_dataframe):
    """Return the DataFrame in the exact format expected by the sentiment analysis pipeline.

    The sentiment step should consume the extracted text for a chosen dimension, not the
    raw manifesto text. The input may already contain a 'text' column from the LLDA
    extraction step or need to be renamed from 'raw_text'.
    """
    if dimension_text_dataframe is None or dimension_text_dataframe.empty:
        raise ValueError("No extracted dimension text available for sentiment analysis.")

    df = dimension_text_dataframe.copy()

    if "text" not in df.columns:
        if "raw_text" in df.columns:
            df = df.rename(columns={"raw_text": "text"})
        else:
            raise ValueError("Extracted dimension data must contain a 'text' column.")

    required = {"party", "year", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fields for sentiment analysis: {sorted(missing)}")

    sentiment_df = (
        df[["party", "year", "text"]]
        .dropna(subset=["text"])
        .copy()
        .reset_index(drop=True)
    )

    sentiment_df = (
        sentiment_df
        .groupby(["party", "year"], as_index=False)["text"]
        .agg(lambda values: " ".join(str(v) for v in values))
    )

    return sentiment_df


def calculate_hellinger_distance(p, q):

	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)

	if p.sum() > 0:
		p = p / p.sum()
	else:
		p = np.zeros_like(p)

	if q.sum() > 0:
		q = q / q.sum()
	else:
		q = np.zeros_like(q)

	return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def build_topic_word_distributions(tmt_model, doc_metadata, categories):
	vocab_size = len(tmt_model.used_vocabs)

	topic_words_by_year = {}
	topic_words_by_year_party = {}

	for topic_id, label in enumerate(categories):

		for doc_idx, doc in enumerate(tmt_model.docs):
			meta = doc_metadata[doc_idx]

			year = meta["year"]
			party = meta["party"]

			year_key = (label, year)
			party_key = (label, year, party)

			if year_key not in topic_words_by_year:
				topic_words_by_year[year_key] = np.zeros(vocab_size)

			if party_key not in topic_words_by_year_party:
				topic_words_by_year_party[party_key] = np.zeros(vocab_size)

			for word_id, assigned_topic in zip(doc.words, doc.topics):
				if assigned_topic == topic_id:
					topic_words_by_year[year_key][word_id] += 1
					topic_words_by_year_party[party_key][word_id] += 1

	return topic_words_by_year, topic_words_by_year_party

def compute_temporal_hellinger_drift(topic_words_by_year):
	rows = []

	labels = sorted(set(label for label, year in topic_words_by_year.keys()))

	for label in labels:
		years = sorted(
			year
			for topic_label, year in topic_words_by_year.keys()
			if topic_label == label
		)

		for i in range(len(years) - 1):
			year_t = years[i]
			year_t1 = years[i + 1]

			dist = calculate_hellinger_distance(
				topic_words_by_year[(label, year_t)],
				topic_words_by_year[(label, year_t1)]
			)

			rows.append({
				"label": label,
				"period": year_t,
				"next_period": year_t1,
				"distance": dist
			})

	return pd.DataFrame(rows)


def compute_party_hellinger_distance(
	topic_words_by_year_party,
	party_a,
	party_b,
	topic=None
):
	rows = []

	labels = sorted(set(
		label
		for label, year, party in topic_words_by_year_party.keys()
	))

	if topic is not None:
		labels = [topic]

	for label in labels:
		years = sorted(set(
			year
			for topic_label, year, party in topic_words_by_year_party.keys()
			if topic_label == label
		))

		for year in years:
			key_a = (label, year, party_a)
			key_b = (label, year, party_b)

			if key_a not in topic_words_by_year_party:
				continue

			if key_b not in topic_words_by_year_party:
				continue

			dist = calculate_hellinger_distance(
				topic_words_by_year_party[key_a],
				topic_words_by_year_party[key_b]
			)

			rows.append({
				"label": label,
				"year": year,
				"party_a": party_a,
				"party_b": party_b,
				"distance": dist
			})

	return pd.DataFrame(rows)


def evaluate_efficiency(tmt_model, doc_metadata, categories):
	npmi_evaluator = tmt.coherence.Coherence(
		tmt_model,
		coherence="c_npmi"
	)

	topic_words_by_year, topic_words_by_year_party = build_topic_word_distributions(
		tmt_model=tmt_model,
		doc_metadata=doc_metadata,
		categories=categories
	)

	temporal_drift_df = compute_temporal_hellinger_drift(
		topic_words_by_year
	)

	npmi_df = pd.DataFrame([
		{
			"label": label,
			"npmi_coherence": npmi_evaluator.get_score(topic_id=topic_id)
		}
		for topic_id, label in enumerate(categories)
	])

	return {
		"npmi_df": npmi_df,
		"temporal_drift_df": temporal_drift_df,
		"topic_words_by_year": topic_words_by_year,
		"topic_words_by_year_party": topic_words_by_year_party
	}
