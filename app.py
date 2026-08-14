import os
import sys
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add project "src" folders to path so the existing modules can be imported
ROOT = os.path.dirname(__file__)
TOPIC_SRC = os.path.join(ROOT, "topic_sentiment_analysis", "src")
UNV_SRC = os.path.join(ROOT, "un_voting_analysis", "src")

for p in (TOPIC_SRC, UNV_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import project functions (the app intentionally uses only the project's methods)
try:
    from exploratory_analysis.exp_ana import NLP
    from topic_analysis import seedwords
    from topic_analysis.llda_functions import (
        run_llda_pipeline_for_seed,
        extract_topic_sentences_by_party,
        build_sentiment_input_from_dimension,
    )
    from topic_analysis.manifesto_preprocessing import preprocess_manifesto
    from sentiment_analysis.country_aliases_generation import generate_wikidata_country_dictionary
    from sentiment_analysis.sentiment_analysis import (
        setup_sentiment_analyzer,
        compute_country_scores,
        plot_country_sentiment,
        compute_sentiment,
    )
except Exception as e:
    st.sidebar.error(f"Could not import topic/sentiment modules: {e}")

try:
    from data_processing import (
        load_raw_votes,
        transform_votes,
        encoder_dictionnaries,
        structure_vote_table,
        build_session_years,
    )
    from model_builder import build_beta_initial_values, build_model
    from inference import run_sampling, check_convergence
    from postprocessing import extract_theta_summary, extract_beta_summary
    from viz import plot_country_trajectories, plot_distance_to_USA, plot_distance_sentiment
except Exception as e:
    st.sidebar.error(f"Could not import UN voting modules: {e}")

st.set_page_config(layout="wide", page_title="Pipeline")
st.title("Topic → Sentiment → UN voting")
st.markdown("This app calls the functions already present in the repository. Follow the steps in each tab.")

# Sidebar: file uploads
st.sidebar.header("Data uploads")
manifestos_file = st.sidebar.file_uploader("Upload manifestos CSV (columns: party, year, raw_text). They are in manifestoData_forApp", type=["csv"] )
votes_file = st.sidebar.file_uploader("Upload UN votes CSV. These are in un_voting_analysis/data", type=["csv"]) 
VOTE_TYPE_OPTIONS = ["All", "NoNukes", "MiddleEast", "HumanRights", "Colonial", "Nuclear", "Economic", "Disarmament"]
vote_type = st.sidebar.selectbox("Restrict UN votes to type", options=VOTE_TYPE_OPTIONS, index=VOTE_TYPE_OPTIONS.index("MiddleEast"))
use_wikidata = st.sidebar.checkbox("Use Wikidata for country synonyms (Do not touch this. Otherwise nothing will work.)", value=True)

# Helper to read uploaded CSV
@st.cache_data
def read_csv(uploaded):
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)


def filter_votes_by_type(votes_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if votes_df is None:
        return None

    df = votes_df.copy()
    if "vote" not in df.columns:
        raise ValueError("The uploaded UN votes file must contain a 'vote' column.")

    df = df[df["vote"].isin({1, 2, 3})].copy()

    if kind == "All":
        return df
    if kind == "NoNukes":
        return df if "nu" not in df.columns else df[df["nu"] == 0]
    if kind == "Important":
        filtered = df.copy()
        if "session" in filtered.columns:
            filtered = filtered[filtered["session"] >= 38]
        if "importantvote" in filtered.columns:
            filtered = filtered[filtered["importantvote"] == 1]
        return filtered
    if kind == "MiddleEast":
        filtered = df.copy()
        if "me" in filtered.columns:
            filtered = filtered[(filtered["me"] == 1) & (filtered["session"] >= 29)]
        return filtered
    if kind == "HumanRights":
        filtered = df.copy()
        if "hr" in filtered.columns:
            filtered = filtered[(filtered["hr"] == 1) & (filtered["session"] >= 25)]
        return filtered
    if kind == "Colonial":
        filtered = df.copy()
        if "co" in filtered.columns:
            filtered = filtered[(filtered["co"] == 1) & (filtered["session"] >= 14)]
        return filtered
    if kind == "Nuclear":
        filtered = df.copy()
        if "nu" in filtered.columns:
            filtered = filtered[(filtered["nu"] == 1) & (filtered["session"] >= 27)]
        return filtered
    if kind == "Economic":
        filtered = df.copy()
        if "ec" in filtered.columns:
            filtered = filtered[(filtered["ec"] == 1) & (filtered["session"] >= 26)]
        return filtered
    if kind == "Disarmament":
        filtered = df.copy()
        if "di" in filtered.columns:
            filtered = filtered[filtered["di"] == 1]
        return filtered

    raise ValueError(f"Unknown vote type '{kind}'.")

manifestos_df = read_csv(manifestos_file)
votes_df = read_csv(votes_file)

# Tabs for pipeline steps
tabs = st.tabs(["Topics", "Sentiment", "UN Voting"])

# -------------------- Topics Tab --------------------
with tabs[0]:
    st.header("Topic extraction")

    if manifestos_df is None:
        st.info("Upload a manifestos CSV in the sidebar or provide a DataFrame with columns: party, year, raw_text.")
    else:
        st.subheader("Preview manifestos")
        st.dataframe(manifestos_df.head())

        st.markdown("---")
        st.subheader("Generate candidate seed words (optional). For each country, a set of seedwords must be generated.")
        gen_seed = st.button("Generate seed words from uploaded manifestos")

        if gen_seed:
            try:
                with st.spinner("Training quick LDA to extract candidate seed words (may take a while)..."):
                    words = seedwords.generate_seed_words(manifestos_df, NLP, seed=42, k=8, top_n_words=150)
                st.write("Candidate words. You should ask a LLM to classify them according to the categories and structure this into a Python dict")
                st.write(words)
            except Exception as e:
                st.error(f"Seed-word generation failed: {e}")

        st.markdown("---")
        st.subheader("Run LLDA (seeded)")
        st.markdown("Provide categories (comma separated) and a Python dict of category -> keywords (seed_words_dict). Use the candidate words to help build the mapping.")

        default_categories = "Social_Affairs,Culture,Education,Economy,Justice,Internal_Affairs,Environment,Foreign_Affairs"
        categories_input = st.session_state.get("llda_categories", default_categories)
        seed_words_text = st.session_state.get("llda_seed_words_text", "")
        seed_val = st.session_state.get("llda_seed", 42)

        with st.form("llda_form"):
            categories_input = st.text_input("Categories", value=categories_input)
            seed_words_text = st.text_area("Seed words mapping (JSON). Example: {\"Foreign_Affairs\": [\"war\", \"diplomacy\"], \"Economy\": [\"trade\", \"tax\"]}", value=seed_words_text)
            seed_val = st.number_input("Random seed", value=int(seed_val), min_value=0)
            run_llda = st.form_submit_button("Run LLDA pipeline")

        if run_llda:
            if manifestos_df is None:
                st.error("Please upload manifestos first.")
            else:
                if not categories_input.strip():
                    st.error("Enter at least one category.")
                else:
                    try:
                        categories = [c.strip() for c in categories_input.split(",") if c.strip()]
                        if seed_words_text.strip():
                            seed_words_dict = json.loads(seed_words_text)
                        else:
                            st.warning("No seed words mapping provided — defaulting to assign all categories to chunks without keywords.")
                            seed_words_dict = {c: [] for c in categories}

                        st.session_state["llda_categories"] = categories_input
                        st.session_state["llda_seed_words_text"] = seed_words_text
                        st.session_state["llda_seed"] = int(seed_val)

                        with st.spinner("Running your LLDA pipeline... this may take a while depending on tomotopy and data size"):
                            result = run_llda_pipeline_for_seed(manifestos_df, NLP, categories, int(seed_val), seed_words_dict)

                        st.session_state["llda_result"] = result
                        st.session_state["llda_categories_list"] = categories
                        st.session_state["llda_seed_words_dict"] = seed_words_dict
                        st.session_state["llda_seed_value"] = int(seed_val)

                        st.success("LLDA finished — showing results preview")
                    except Exception as e:
                        st.error(f"LLDA pipeline failed: {e}")

        if "llda_result" in st.session_state:
            result = st.session_state["llda_result"]
            topic_df = result.get("topic_df")
            st.subheader("Topic assignments (preview)")
            st.dataframe(topic_df.head())

            st.subheader("Extract text for a specific dimension")
            default_dimension = st.session_state.get("dimension_name", "Foreign_Affairs")
            with st.form("dimension_form"):
                dimension_name = st.text_input("Dimension label to extract for sentiment", value=default_dimension)
                extract_dim = st.form_submit_button("Extract dimension text")

            if extract_dim:
                try:
                    categories = st.session_state.get("llda_categories_list", [])
                    seed_words_dict = st.session_state.get("llda_seed_words_dict", {})
                    seed_value = st.session_state.get("llda_seed_value", 42)
                    extracted_dimension_df = extract_topic_sentences_by_party(
                        manifestos_dataframe=manifestos_df,
                        nlp=NLP,
                        categories=categories,
                        seed_words_dict=seed_words_dict,
                        topic=dimension_name,
                        seed=int(seed_value),
                    )
                    st.session_state["dimension_name"] = dimension_name
                    st.session_state["dimension_text_df"] = extracted_dimension_df
                    st.success(f"{dimension_name} text extracted succesfully !")
                    st.dataframe(extracted_dimension_df.head())
                except Exception as e:
                    st.error(f"Dimension extraction failed: {e}")

            if "dimension_text_df" in st.session_state:
                st.dataframe(st.session_state["dimension_text_df"].head())

            st.subheader("Evaluation report (keys)")
            st.write(list(result.get("evaluation_report", {}).keys()))

# -------------------- Sentiment Tab --------------------
with tabs[1]:
    st.header("Sentiment analysis: Extract the sentiment of a country towards others")

    if manifestos_df is None:
        st.info("Upload manifestos CSV in the sidebar to run sentiment analysis.")
    else:
        st.subheader("Prepare sentiment analyzer")
        if use_wikidata:
            if "wikidata_country_synonyms" not in st.session_state:
                st.write("Generating country synonyms from Wikidata (internet required)")
                try:
                    with st.spinner("Querying Wikidata and building patterns..."):
                        st.session_state["wikidata_country_synonyms"] = generate_wikidata_country_dictionary()
                    st.success("Country synonyms loaded")
                except Exception as e:
                    st.error(f"Wikidata country dictionary failed: {e}")
                    st.session_state["wikidata_country_synonyms"] = {}
            country_synonyms = st.session_state["wikidata_country_synonyms"]
        else:
            st.write("Wikidata not selected — using empty country synonyms. For targeted sentiment you should enable Wikidata or provide a mapping.")
            country_synonyms = {}

        if st.button("Setup sentiment analyzer"):
            try:
                analyzer = setup_sentiment_analyzer(NLP, country_synonyms)
                st.session_state["analyzer"] = analyzer
                st.success("Sentiment analyzer ready")
            except Exception as e:
                st.error(f"Failed to setup analyzer: {e}")
                st.session_state["analyzer"] = None
                analyzer = None

        if "analyzer" in st.session_state and st.session_state["analyzer"] is not None:
            analyzer = st.session_state["analyzer"]

        if st.button("Compute targeted country sentiment for selected dimension text"):
            try:
                if "analyzer" not in st.session_state or st.session_state["analyzer"] is None:
                    st.warning("Analyzer not initialised — attempting to setup with current configuration")
                    analyzer = setup_sentiment_analyzer(NLP, country_synonyms)
                    st.session_state["analyzer"] = analyzer

                if 'dimension_text_df' in st.session_state and not st.session_state['dimension_text_df'].empty:
                    sentiment_input = build_sentiment_input_from_dimension(st.session_state['dimension_text_df'])
                    st.info("Using the extracted text for the selected dimension, not the raw manifesto text.")
                    st.dataframe(sentiment_input.head())
                else:
                    sentiment_input = manifestos_df[["party", "year", "raw_text"]].rename(columns={"raw_text": "text"})
                    st.warning("No extracted dimension text found; falling back to the raw manifesto text.")

                with st.spinner("Computing targeted sentiment on the extracted dimension text (uses nlp.pipe) — may take time"):
                    scored = compute_country_scores(sentiment_input, NLP, st.session_state["analyzer"], country_synonyms, batch_size=16)
                st.session_state["scored_df"] = scored

                st.success("Sentiment computed — preview")
                st.dataframe(scored.head())

            except Exception as e:
                st.error(f"Sentiment computation failed: {e}")

        if "scored_df" in st.session_state:
            scored = st.session_state["scored_df"]
            st.subheader("Plot sentiment for a party/year")
            with st.form("plot_sentiment_form"):
                party = st.text_input("Party (exact string from 'party' column)", value=st.session_state.get("plot_party", ""))
                year = st.number_input("Year", value=int(st.session_state.get("plot_year", manifestos_df['year'].min())) if 'year' in manifestos_df.columns else 0)
                plot_sentiment_submit = st.form_submit_button("Plot party/year sentiment")

            if plot_sentiment_submit:
                    try:
                        st.session_state["plot_party"] = party
                        st.session_state["plot_year"] = int(year)
                        fig = plot_country_sentiment(scored, party, int(year), show=False)
                        if fig is not None:
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Plot failed: {e}")
# -------------------- UN Voting Tab --------------------
with tabs[2]:
    st.header("UN voting analysis (uses un_voting_analysis functions)")

    st.markdown("Upload the standard UN votes CSV used in the project. Can be found in un_voting_analysis/data/raw")

    if votes_df is None:
        st.info("Upload votes CSV in the sidebar to run UN voting analysis.")
    else:
        st.subheader("Preview votes")
        st.dataframe(votes_df.head())

        st.markdown("---")
        st.subheader("Prepare votes for modeling")
        st.caption(f"Current filter: {vote_type}")
        if st.button("Transform & encode votes"):
            try:
                filtered_votes_df = filter_votes_by_type(votes_df, vote_type)
                st.info(f"Using {len(filtered_votes_df)} rows after the '{vote_type}' filter.")
                matrix = transform_votes(filtered_votes_df)
                country_to_idx, session_to_idx, rcid_to_idx = encoder_dictionnaries(matrix)
                structured = structure_vote_table(matrix, country_to_idx, session_to_idx, rcid_to_idx)
                st.success("Votes transformed and encoded — preview")
                st.dataframe(structured.head())

                st.session_state['raw_votes_df'] = matrix
                st.session_state['votes_matrix'] = structured
                st.session_state['country_to_idx'] = country_to_idx
                st.session_state['session_to_idx'] = session_to_idx
                st.session_state['rcid_to_idx'] = rcid_to_idx
            except Exception as e:
                st.error(f"Vote preparation failed: {e}")

        if 'votes_matrix' in st.session_state:
            structured = st.session_state['votes_matrix']
            matrix = st.session_state['raw_votes_df']
            # Prepare arrays for the model
            country_idx = structured['country_idx'].values
            session_idx = structured['session_idx'].values
            resolution_idx = structured['resolution_idx'].values
            observed_vote = structured['vote_ordinal'].values

            num_countries = len(st.session_state['country_to_idx'])
            num_sessions = len(st.session_state['session_to_idx'])
            num_resolutions = len(st.session_state['rcid_to_idx'])

            st.write(f"Countries: {num_countries}, Sessions: {num_sessions}, Resolutions: {num_resolutions}")

            st.markdown("---")
            st.subheader("Build model & run sampling")
            draws = st.number_input("MCMC draws", value=50, min_value=10)
            tune = st.number_input("MCMC tune", value=10, min_value=0)
            target_accept = st.slider("Target accept", 0.5, 0.99, 0.9)

            if st.button("Build & sample model"):
                try:
                    beta_init = build_beta_initial_values(matrix, st.session_state['rcid_to_idx'], num_resolutions)
                    model = build_model(country_idx, session_idx, resolution_idx, observed_vote, num_countries, num_sessions, num_resolutions, st.session_state['country_to_idx'])

                    with st.spinner("Running MCMC sampling (this can be slow). Use small draws for quick tests."):
                        idata = run_sampling(model, beta_init, draws=int(draws), tune=int(tune), chains=3
                        ,target_accept=float(target_accept))

                    # Postprocessing
                    session_years = build_session_years(structured)
                    theta_summary = extract_theta_summary(idata, st.session_state['country_to_idx'], session_years)
                    theta_summary = theta_summary.copy()
                    theta_summary["vote_type"] = vote_type

                    st.success("Sampling and postprocessing finished")
                    st.subheader("Theta summary preview")
                    st.caption(f"Results computed with vote type: {vote_type}")
                    st.dataframe(theta_summary.head())

                    st.session_state['theta_summary'] = theta_summary
                    st.session_state['theta_vote_type'] = vote_type

                except Exception as e:
                    st.error(f"Model or sampling failed: {e}")

        if 'theta_summary' in st.session_state:
            theta_summary = st.session_state['theta_summary']
            st.subheader("UN Voting Visualizations")
            st.caption(f"Downloaded results will be labeled with vote_type='{st.session_state.get('theta_vote_type', vote_type)}'.")
            csv_bytes = theta_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download theta_summary as CSV",
                data=csv_bytes,
                file_name=f"theta_summary_{st.session_state.get('theta_vote_type', vote_type).lower()}.csv",
                mime="text/csv",
            )
            countries_input = st.text_input("Enter comma-separated country codes to plot (e.g. USA, AUS)")
            if st.button("Plot trajectories"):
                countries = [c.strip() for c in countries_input.split(",") if c.strip()]
                if countries:
                    fig = plot_country_trajectories(theta_summary, countries, conf_intervals=False)
                    st.pyplot(fig)
                else:
                    st.error("Provide at least one country code")

            if st.button("Plot distance to USA for listed countries"):
                try:
                    countries = [c.strip() for c in countries_input.split(",") if c.strip()]
                    fig, df_dist = plot_distance_to_USA(theta_summary, countries, return_figure=True)
                    st.pyplot(fig)
                    st.dataframe(df_dist.head())
                except Exception as e:
                    st.error(f"Distance plot failed: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("This UI calls only the functions present in the repository — no refactor performed. Some operations (LLDA, tomotopy, PyMC sampling, and Wikidata lookups) may require heavy dependencies and internet access.")
