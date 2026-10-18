# Classification Studio

An end-to-end classifier for tabular data with **inputs A–G and label H**. You upload labeled data. The app explores it,
cleans it, picks a validated model, and scores new files live. An on-demand **AI agent** explains each individual
prediction.

* **Tab 1 – Explore, preprocess & model:** EDA, the cleaning log, before/after views of every transformation, the
  split strategy, a comparison of 6 tuned model families plus an ensemble, and test-set results.
* **Tab 2 – Predict & explain:** upload the hold-out set (or any A–G file). The saved pipeline cleans, encodes and
  scores it right away. Tick **💬 Explain** on any row to ask the agent why the model predicted it.

## Run it

Needs Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                 # opens http://localhost:8501
```

The trained model ships in `artifacts/`, so the dashboard opens ready to use. If the model is missing or can't be
loaded, the app retrains itself from the newest file in `data/` (about 1 minute).

### Train on your own data

To change the shared model, run:

```bash
python train.py data/<your_file>.xlsx
```

You can also upload a labeled CSV/Excel file in **Tab 1 → "Train on a labeled dataset"**. That model is used only in
your own browser session, so visitors to a public link can't replace the shared model.

Training writes `artifacts/model.joblib` and the 10% hold-out set, `artifacts/holdout_test.csv` (columns A–G). That
hold-out file is what you upload in Tab 2. A copy with labels, `holdout_test_labeled.csv`, makes Tab 2 also report
accuracy. `data/synthetic_demo.csv` is a deliberately messy synthetic dataset with the same shape, used to show the
pipeline works on any data.

### The AI explainer: no key needed

| Option | Setup | Key? |
|---|---|---|
| Offline explainer (default fallback) | nothing | no |
| **Open-source LLM, local** | install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b` (auto-detected) | no |
| OpenAI / Anthropic / any OpenAI-compatible API | set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, or paste a key in the sidebar | yourself |

Keys are never stored in code or on disk. A key typed into the sidebar lives only in that browser session. You can
also use `.streamlit/secrets.toml`; don't ship that file.

### Verify

```bash
python test_smoke.py     # trains a small model, then tries messy uploads, Shapley additivity, and the agent loop (~10 s)
```

### Share a public link

**Quick and temporary.** This computer must stay on and awake, and the URL changes on every restart:

```bash
brew install cloudflared                                   # once
streamlit run app.py --server.port 8502 --server.address 127.0.0.1 &
cloudflared tunnel --url http://127.0.0.1:8502             # prints https://<random>.trycloudflare.com
```

**Permanent and free.** Push this folder to a GitHub repo. On [share.streamlit.io](https://share.streamlit.io), choose
"Create app", pick the repo and `app.py`, set Python 3.13 under Advanced settings, and deploy.

Anyone with a public link can see everything in Tab 1, including sample rows, and can download the hold-out set. Only
put synthetic or approved data behind a public link.

## Project structure

| File | Role |
|---|---|
| `pipeline.py` | reading files, schema inference, cleaning, the sklearn pipeline, model search, evaluation, inference |
| `agent.py` | Shapley explanations, agent tools, the LLM tool-calling loop, the offline fallback |
| `app.py` | the Streamlit dashboard (both tabs) |
| `train.py` | command-line training |
| `test_smoke.py` | end-to-end checks |
| `data/` | labeled input data |
| `artifacts/` | trained model bundle and the hold-out files |

## Design decisions (what to ask me about)

**Cleaning.** A column is numeric if at least 90% of its values parse as numbers; otherwise it is categorical.
Constant, row-index and ID-like columns are dropped. Numbers are parsed tolerantly (`"1,234"` → 1234). Placeholders
such as `?` and `N/A` become missing. Category spelling is normalised (`" WEST "` = `"West"`). Exact duplicate rows
are removed **before** splitting, so the same row can't land in both train and test. Every step is logged and shown
in Tab 1.

**Preprocessing.** All of it sits inside one sklearn `Pipeline`, fitted on training rows only, so Tab 2 applies
exactly the same transformations.
* Numeric: winsorise to the 1st–99th percentile (outliers), median-impute, standardise, then Yeo-Johnson (de-skew).
* Missing-value flags: missingness can carry signal, so it is kept as its own feature.
* Categorical: missing becomes its own level; one-hot encoding groups levels under 1% into "infrequent". Categories
  never seen in training map to that same bucket, so unseen values never crash prediction.

**Split.**
* A stratified 10% test set is held out first (seed 42). It is never used for tuning or model choice, and it is
  scored exactly once.
* The other 90% goes through stratified 5-fold CV. Every fold re-fits the preprocessing, so no information leaks.
  Five folds give a mean ± std; a single validation slice of about 1k rows would be noisy and waste data.

**Models.**
* Six model families: logistic regression, k-NN, random forest, extra trees, HistGradientBoosting and an MLP neural
  net, plus a soft-voting ensemble of the top 3.
* Each is tuned by randomised search and judged on CV accuracy, with balanced accuracy and macro-F1 alongside. A
  majority-class baseline is the bar to beat.
* The winner is refit on the full 90%. Reported: accuracy, balanced accuracy, macro-F1, ROC-AUC, log-loss, confusion
  matrix, ROC curve, calibration ("when it's confident, is it right?"), permutation importance and partial dependence.

**Robust inference (Tab 2).**
* Reads CSV/TSV/Excel with any delimiter, UTF-8 or Latin-1.
* Matches columns case-insensitively and in any order, ignores extra columns, and imputes missing columns with a
  warning.
* Files without a header row are mapped by position, with a warning.
* Bad values, unseen categories and out-of-range numbers each produce a clear warning instead of a crash.
* If column H is present, the predictions are scored against it.

**Agent architecture.**
* Each Explain click runs one investigation for that row only. Nothing is precomputed.
* The tools are plain Python over the trained model: exact interventional **Shapley values** (all 2⁷ feature
  coalitions, so contributions add up exactly from the baseline to the prediction); training-data context
  (percentiles, per-class medians, category mixes); a **counterfactual search** for the smallest single-feature change
  that flips the outcome; what-if re-scoring; and a model card.
* An LLM chooses which tools to call and writes the explanation, quoting only numbers the tools returned. It works
  with any OpenAI-compatible endpoint, so a local open-source model, OpenAI and Anthropic all go through the same code.
* With no LLM, or if a call fails, the same tools fill a template, so **Explain always works**.
