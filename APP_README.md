# Political Manifesto Analysis and UN Voting App

This repository contains a Streamlit application that combines two research pipelines:

1. Topic extraction and sentiment analysis from political manifestos.
2. UN General Assembly vote analysis using a Bayesian latent-position model.

The app is designed for users who want to run the analysis without manually writing Python code. It is meant to be accessible, but it still depends on the project data being structured correctly.

This guide explains:

- how to install the project,
- what data is needed,
- how to launch the app,
- how to use each tab,
- what files are created and downloaded,
- what to do if the app reports an error.

---


# Step-by-step workflow for a new user

### Step 1: Install dependencies

Run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Launch the app. You should launch this in a bash terminal.

```bash
streamlit run app.py
```

### Step 3: Upload manifesto data

In the sidebar:

- upload a CSV with `party`, `year`, and `raw_text`. These csv can be found in the folder 'manifestoData_forApp' or computed for each country by using the notebook fecthing_manifestoData_forApp
- or use the bundled manifesto data from the project folders

### Step 4: Use the Topics tab

- choose categories such as `Foreign_Affairs`, `Economy`, `Justice`, etc.
- add a seed-word mapping in JSON form if needed. To do this you should use an LLM and tell him to make the classification based on the seed_words that the algo generated on the interface.
- click “Run LLDA pipeline”
- then extract the text for the dimension of interest

Example seed mapping:

```json
{
  "Foreign_Affairs": ["war", "diplomacy", "peace", "security"],
  "Economy": ["trade", "tax", "growth", "market"]
}
```

### Step 5: Use the Sentiment tab

- make sure a sentiment analyzer is set up
- enable Wikidata if you want country matching
- click “Compute targeted country sentiment for selected dimension text”
- this uses the extracted dimension text rather than the full raw manifesto if available

### Step 6: Upload UN voting data

- in the sidebar, upload the UN votes CSV
- choose the vote type filter you want
- click “Transform & encode votes”

### Step 7: Run the UN model

- choose the number of draws and tuning iterations
- click “Build & sample model”
- wait for the model to finish sampling

### Step 8: Download the results

After model fitting, the app provides a CSV export for the `theta_summary` results, automatically labeled with the vote type used.

---

## 6. Recommended usage patterns

### For manifesto analysis only

Use:

- the Topics tab,
- the Sentiment tab,
- a manifesto CSV with `party`, `year`, and `raw_text`.

### For UN voting analysis only

Use:

- the UN Voting tab,
- a raw UN votes CSV or a normalized version,
- a selected vote-type filter.

### For integrated analysis

Use both parts together:

- extract a dimension from the manifestos,
- compute sentiment on that dimension,
- compare the textual results with the final latent country positions from the UN voting model.

This combination is the main purpose of the project.

---

## 7. Output files and what they contain

### Manifesto-related outputs

These are not always exported as separate files by the app, but the app generates tables in memory for:

- topic assignments,
- extracted dimension text,
- sentiment scores,
- party/year plots.

### UN voting outputs

The app produces:

- a transformed and encoded vote matrix,
- posterior summary tables like `theta_summary`,
- country trajectories and distance plots,
- a CSV download of the final summary labeled by `vote_type`.

---

## 8. Common issues and how to fix them

### Error: spaCy model not found

Run:

```bash
python -m spacy download en_core_web_sm
```

### Error: missing columns in CSV

Check that your CSV has the required columns:

- manifestos: `party`, `year`, `raw_text`
- UN votes: `Country` or `country_code`, `vote`, `session`, `rcid`, `year`

### Error: empty data after applying a vote filter

This can happen if the chosen vote type produces no rows. Try a broader filter such as `All` or `Important`.

### Error: model or sampling failed

This often means the data is malformed or incomplete. Check:

- whether the vote column uses valid codes,
- whether country names are present,
- whether the dataset has enough rows for the selected subset.

### Warning: plotting backend issue

This project uses Streamlit, which expects figures to be rendered through `st.pyplot(...)` rather than `plt.show()`. The app is set up to do this correctly.

---

## 9. Recommended starting point

If you are not an expert in the project, the easiest workflow is:

1. install dependencies,
2. launch `streamlit run app.py`,
3. upload a manifestos CSV,
4. use the Topics tab to extract a dimension,
5. run sentiment on that extracted text,
6. upload the UN votes CSV and select a vote type,
7. run the model,
8. download the resulting `theta_summary` CSV.

This is the cleanest and most reproducible way to use the app.

---

## 10. Summary

This project is a practical interface for analyzing political positions from two complementary sources:

- what political parties say in manifestos,
- how countries vote in the UN.

The app is designed to make the workflow accessible while keeping the underlying analysis logic intact.

If you follow the structure above, you should be able to run the project without advanced Python knowledge.

