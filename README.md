# Classification Studio

**Live demo: [classification-studio-sihanwang.streamlit.app](https://classification-studio-sihanwang.streamlit.app/)**
(running on a synthetic demo dataset)

An end-to-end classifier for tabular data with **inputs A–G and label H**. You upload labeled data. The app explores it,
cleans it, picks a validated model, and scores new files live. An on-demand **AI agent** explains each individual
prediction.

* **Tab 1 – Explore, preprocess & model:** an overview of the whole analysis, a browsable view of the training data
  (as uploaded vs. after cleaning, with every fixed cell highlighted), EDA, the cleaning log, before/after views of every transformation, the
  split strategy, a comparison of 6 tuned model families plus an ensemble, and test-set results.
* **Tab 2 – Predict & explain:** upload the hold-out set (or any A–G file). The saved pipeline cleans, encodes and
  scores it right away. Tick **💬 Explain** on any row to ask the agent why the model predicted it, then ask it
  follow-up questions ("what if A were 30?", "what would flip it?").

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

You can also upload a labeled CSV/Excel file in **Tab 1 → "Train on a labeled dataset"**. Run locally, that replaces the
saved model too. On the hosted demo (or with `PUBLIC_DEMO=1`), the model stays in the visitor's own browser session, so
visitors can't replace the shared one.

Training writes `artifacts/model.joblib` and the 10% hold-out set, `artifacts/holdout_test.csv` (columns A–G). That
hold-out file is what you upload in Tab 2. A copy with labels, `holdout_test_labeled.csv`, makes Tab 2 also report
accuracy. `data/synthetic_demo.csv` is a deliberately messy synthetic dataset with the same shape, used to show the
pipeline works on any data.

### The AI agent: a local open-source model via Ollama

The agent runs on [Ollama](https://ollama.com) with the open-source `qwen2.5:7b` model. No API key is needed, and data
never leaves the machine. Install Ollama, then:

```bash
ollama pull qwen2.5:7b      # once, about 4.7 GB
```

The app connects to Ollama by itself; there is nothing to configure in the UI. If Ollama isn't running, Explain uses an
offline template built on the same analysis, and the chat still answers what-if and "what would flip it?" questions.

**Hosted website, always on:** Streamlit Cloud can't run Ollama itself, so point the app at Ollama's hosted API. In
the app's **Settings → Secrets** on share.streamlit.io, add the lines below. The key stays in Streamlit's secret store,
never in the repo.

```toml
OLLAMA_HOST = "https://ollama.com"
OLLAMA_API_KEY = "<your key from ollama.com/settings/keys>"
OLLAMA_MODEL = "gpt-oss:120b"   # or any tool-capable model from ollama.com's cloud list
```

Backend settings, all optional, as environment variables or in `.streamlit/secrets.toml`:
* `OLLAMA_HOST`: default `http://localhost:11434`;
* `OLLAMA_MODEL`: default `qwen2.5:7b`, otherwise the best installed chat model;
* `OLLAMA_API_KEY`: only for a remote Ollama server that requires one.

### Verify

```bash
python test_smoke.py     # trains a small model, then tries messy uploads, Shapley additivity, and the agent loop (~10 s)
```

### Share a public link

**Quick and temporary.** This computer must stay on and awake, and the URL changes on every restart:

```bash
brew install cloudflared                                   # once
PUBLIC_DEMO=1 streamlit run app.py --server.port 8502 --server.address 127.0.0.1 &
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
* A local open-source LLM (Ollama, `qwen2.5:7b`) chooses which tools to call and writes the explanation, quoting only
  numbers the tools returned.
* **Ask the agent:** under each explanation, a chat box lets you ask follow-up questions about that row. The agent
  answers by calling the same tools, for example re-scoring "what if A were 30?".
* With no LLM, or if a call fails, the same tools fill a template, so **Explain always works**.
