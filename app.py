"""Interactive dashboard.   Run:  streamlit run app.py"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score

import agent
import pipeline as pl
from agent import _fmt as fmt

st.set_page_config(page_title="Classification Studio", page_icon="🧭", layout="wide")
try:  # optional .streamlit/secrets.toml -> env vars; keys never live in code
    os.environ.update({k: str(v) for k, v in st.secrets.items() if k.endswith("_API_KEY")})
except Exception:
    pass

# Models trained from the UI live in a per-session temp dir, so a visitor can never overwrite the shared model
# in artifacts/ (that one changes only via `python train.py`).
ART = Path(st.session_state.get("model_dir", pl.ART))
if not (ART / "model.joblib").exists():  # session model vanished (temp cleanup) -> back to the shared one
    ART = pl.ART
    st.session_state.pop("model_dir", None)
MODEL = ART / "model.joblib"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = [[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#d03b3b"]]
MUTED, GRID = "#898781", "rgba(137,135,129,0.2)"
FILE_TYPES = ["csv", "txt", "tsv", "xlsx", "xlsm"]


def show(fig, height=340, key=None, **layout):
    base = dict(height=height, margin=dict(l=8, r=8, t=76 if "title" in layout else 40, b=8), bargap=0.15,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""))
    fig.update_layout(**(base | layout))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, key=key, config={"displaylogo": False})


# ----------------------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="Loading model...", max_entries=20)
def load_bundle(path, mtime):  # mtime busts the cache after retraining
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def cleaned(_b, key):
    s = _b["schema"]
    c = pl.Cleaner(s["numeric"], s["categorical"]).transform(_b["data"])
    c[_b["label"]] = _b["data"][_b["label"]].astype(str)
    c["split"] = _b["data"]["split"]
    return c


@st.cache_data(show_spinner=False)
def mutual_info(_b, key):
    c, s = cleaned(_b, key), _b["schema"]
    X = pd.concat([c[s["numeric"]].fillna(c[s["numeric"]].median()),
                   c[s["categorical"]].fillna("missing").apply(lambda col: col.astype("category").cat.codes)], axis=1)
    mi = mutual_info_classif(X, c[_b["label"]], discrete_features=[f in s["categorical"] for f in X.columns], random_state=0)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def transformed(_b, key):
    enc = _b["prep"].named_steps["encode"]
    names = [n.replace("num__", "").replace("cat__", "").replace("missing_flag__missingindicator_", "missing_")
             for n in enc.get_feature_names_out()]
    train = _b["data"][_b["data"].split == "train"]
    return pd.DataFrame(_b["prep"].transform(train), columns=names, index=train.index)


@st.cache_data(show_spinner=False)
def partial_dependence(_b, key, f):
    c, d = cleaned(_b, key), _b["data"]
    rows = d[d.split == "train"].sample(min(300, (d.split == "train").sum()), random_state=0)[_b["all_features"]]
    grid = (np.unique(np.nanquantile(c[f].dropna(), np.linspace(0.02, 0.98, 25))) if f in _b["schema"]["numeric"]
            else c[f].fillna("missing").value_counts().index[:15].tolist())
    P = _b["pipeline"].predict_proba(pd.concat([rows.assign(**{f: g}) for g in grid]))
    return grid, P.reshape(len(grid), len(rows), -1).mean(1)


@st.cache_data(show_spinner="Running the preprocessing + model pipeline...", max_entries=20)
def process(_b, key, name, data):
    t0 = time.perf_counter()
    X, y, issues = pl.align(pl.read_table(data, name), _b)
    out, proba = pl.predict(_b, X)
    return X, y, issues, out, proba, time.perf_counter() - t0


@st.cache_data(ttl=30, show_spinner=False)
def detect_llm():
    return agent.auto_llm()


def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())


# ----------------------------------------------------------------------------- training entry point
def trainer(expanded):
    with st.expander("⬆️ Train on a labeled dataset (columns A-G + label H)", expanded=expanded):
        st.caption("Any CSV/Excel with the A-G inputs and the H label. The app infers column types, cleans, splits "
                   "90/10, tunes and compares 6 model families + an ensemble (~1 min for 6k rows). The new model is used "
                   "in **your browser session only**; the shared model is unchanged.")
        f = st.file_uploader("Labeled training file", type=FILE_TYPES, key="train_upload")
        if f is not None and st.button(f"Train on {f.name}", type="primary"):
            bar = st.progress(0.0, "Reading file...")
            out = tempfile.mkdtemp(prefix="session_model_")  # ponytail: never cleaned up; fine for a demo server
            try:
                pl.train(pl.read_table(f.getvalue(), f.name), source=f.name, out_dir=out,
                         progress=lambda frac, msg: bar.progress(min(float(frac), 1.0), msg))
            except Exception as e:
                st.error(f"Training failed: {e}")
                return
            st.session_state.model_dir = out
            st.rerun()


def auto_train(reason):
    """No usable saved model: retrain from the newest labeled file in data/ instead of failing."""
    files = sorted((f for f in (pl.ROOT / "data").glob("*") if f.suffix.lower()[1:] in FILE_TYPES), key=lambda f: f.stat().st_mtime)
    if not files:
        return
    st.info(f"{reason} Retraining from `data/{files[-1].name}` (one-off, about a minute)...")
    bar = st.progress(0.0, "Reading file...")
    pl.train(pl.read_table(files[-1].read_bytes(), files[-1].name), source=files[-1].name,
             progress=lambda frac, msg: bar.progress(min(float(frac), 1.0), msg))
    st.rerun()


try:
    b = load_bundle(str(MODEL), MODEL.stat().st_mtime) if MODEL.exists() else None
except Exception as e:  # e.g. artifact pickled with different library versions
    b = None
    auto_train(f"The saved model could not be loaded ({type(e).__name__}).")
if b is None and not MODEL.exists():
    auto_train("No trained model found.")

# ----------------------------------------------------------------------------- sidebar: model + agent settings
with st.sidebar:
    st.subheader("Model")
    if b:
        st.markdown(f"**{b['model_name']}**  \ntrained on `{b['source']}`  \n{b['trained_at'].replace('T', ' ')}")
        st.metric("Test-set accuracy", f"{b['test']['accuracy']:.1%}",
                  f"{100 * (b['test']['accuracy'] - b['test']['baseline_accuracy']):+.1f} pts vs majority baseline")
    if "model_dir" in st.session_state:
        st.caption("Session model - trained from your upload, visible only to you.")
        if st.button("↩ Back to the shared model"):
            del st.session_state["model_dir"]
            st.rerun()
    st.divider()
    st.subheader("🤖 Explainer agent")
    choice = st.selectbox("LLM", ["Auto-detect", *agent.PROVIDERS, "Offline (no LLM)"],
                          help="Auto-detect: OPENAI_API_KEY / ANTHROPIC_API_KEY env var, else a running Ollama, else offline.")
    if choice == "Auto-detect":
        llm = detect_llm()
    elif choice == "Offline (no LLM)":
        llm = None
    else:
        p = agent.PROVIDERS[choice]
        needs_key = choice in ("OpenAI", "Anthropic (Claude)")
        base = p["base_url"] if needs_key else st.text_input("Base URL", p["base_url"])
        default_model = p["model"]
        if choice.startswith("Ollama"):
            found = agent.ollama_models()
            default_model = next((m for pref in agent.OLLAMA_PREFS for m in found if m.startswith(pref)), found[0] if found else p["model"])
        model = st.text_input("Model", default_model)
        key = st.text_input("API key", type="password", help="Held in memory for this browser session only; never saved.") if p["env"] else ""
        key = key or os.environ.get(p["env"] or "", "")
        llm = {"provider": choice, "base_url": base, "model": model, "api_key": key} if (key or not needs_key) and model and base else None
    if llm:
        st.success(f"Using **{llm['provider']}** · `{llm['model']}`", icon="🟢")
    else:
        st.info("No LLM configured - the **offline explainer** runs the same tools and fills a template. "
                "For an open-source LLM with no key: install [Ollama](https://ollama.com) and `ollama pull qwen2.5:7b`.", icon="⚪")

st.title("🧭 Classification Studio")
st.caption("Labeled data in -> exploration -> cleaning -> validated model -> live predictions with an on-demand AI explainer.")
if b is None:
    st.info("No trained model yet. Upload a labeled dataset to train one - or run `python train.py data/<file>`.")
    trainer(expanded=True)
    st.stop()

KEY, LABEL = f"{MODEL}|{b['trained_at']}", b["label"]
NUM, CAT, FEATS = b["schema"]["numeric"], b["schema"]["categorical"], b["features"]
CLASSES = [str(c) for c in b["classes"]]
CMAP = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(CLASSES)}
FOCUS = CLASSES[1] if len(CLASSES) == 2 else min(CLASSES, key=lambda c: b["class_mix"].get(c, 0))  # class used in rate charts

tab1, tab2 = st.tabs(["📊 1 · Explore, preprocess & model", "🔮 2 · Predict & explain"])

# ============================================================================= TAB 1
with tab1:
    st.markdown(f"Current model trained on **{b['source']}** · {b['trained_at'].replace('T', ' ')} · "
                f"{b.get('train_seconds', 0):.0f}s end-to-end")
    trainer(expanded=False)
    c = cleaned(b, KEY)
    y = c[LABEL]
    mi = mutual_info(b, KEY)
    eda, prep, split, model_tab = st.tabs(["Data exploration", "Cleaning & preprocessing", "Split strategy", "Training & evaluation"])

    # ------------------------------------------------------------------ exploration
    with eda:
        k = st.columns(5)
        k[0].metric("Rows (after de-dup)", f"{len(c):,}")
        k[1].metric("Features used", len(FEATS), help=f"{len(NUM)} numeric ({', '.join(NUM)}) · {len(CAT)} categorical ({', '.join(CAT)})")
        k[2].metric("Missing cells", f"{c[FEATS].isna().mean().mean():.1%}")
        k[3].metric("Classes", y.nunique())
        k[4].metric("Majority-class share", f"{y.value_counts(normalize=True).iloc[0]:.1%}", help="Accuracy of always guessing the most common class - the bar to beat.")

        share = y.value_counts(normalize=True)
        notes = [f"**Class balance:** " + ", ".join(f"`{cl}` {v:.1%}" for cl, v in share.items())
                 + (" - imbalanced, so balanced accuracy and macro-F1 are tracked next to accuracy." if share.min() < 0.2
                    else " - balanced enough that accuracy is a fair headline metric.")]
        miss = c[FEATS].isna().mean()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            f0 = miss.index[0]
            r_m, r_p = (y[c[f0].isna()] == FOCUS).mean(), (y[c[f0].notna()] == FOCUS).mean()
            notes.append("**Missing values** in " + ", ".join(f"`{f}` ({v:.1%})" for f, v in miss.items())
                         + (f". Missingness is *informative*: `{FOCUS}` rate is {r_m:.0%} when `{f0}` is missing vs {r_p:.0%} "
                            "when present -> kept as a missing-indicator feature instead of silently imputing." if abs(r_m - r_p) > 0.05 else "."))
        weak = mi[mi < 0.005].index.tolist()
        notes.append(f"**Strongest single-feature signal:** `{mi.index[0]}`" + (f", then `{mi.index[1]}`" if len(mi) > 1 else "")
                     + " (mutual information)." + (f" {', '.join(f'`{w}`' for w in weak)} carry almost no signal on their own." if weak else ""))
        if len(NUM) >= 2:
            cr = c[NUM].corr("spearman").abs().where(np.triu(np.ones((len(NUM), len(NUM)), bool), 1)).stack()
            notes.append(f"**Correlation:** strongest numeric pair `{cr.idxmax()[0]}`-`{cr.idxmax()[1]}` (|ρ| = {cr.max():.2f})"
                         + (" -> redundant information; regularised / tree models handle this." if cr.max() > 0.7 else " -> little redundancy between inputs."))
        if NUM:
            o = pd.Series({f: iqr_outliers(c[f].dropna()) for f in NUM}).sort_values(ascending=False)
            notes.append(f"**Outliers:** `{o.index[0]}` has {o.iloc[0]:,} values beyond 1.5×IQR"
                         + (f", `{o.index[1]}` {o.iloc[1]:,}" if len(o) > 1 else "") + " -> winsorised to the 1st-99th percentile, then Yeo-Johnson de-skewed.")
        log = b["log"]
        fixes = log[log.Step.str.startswith(("Coerced", "Merged", "Removed")) & (log.Affected > 0)]
        if len(fixes):
            notes.append("**Data quality fixes:** " + "; ".join(f"{s} ({a:,})" for s, a in zip(fixes.Step, fixes.Affected)) + ".")
        with st.container(border=True):
            st.markdown("#### Key findings\n" + "\n".join(f"- {n}" for n in notes))

        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### Class balance")
            vc = y.value_counts().reindex(CLASSES)
            fig = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_map=CMAP, text=[f"{v:,} ({v / vc.sum():.0%})" for v in vc.values],
                         labels={"x": LABEL, "y": "rows"})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(type="category")
            show(fig, 320, showlegend=False)
        with right:
            st.markdown("#### Data quality by column")
            rows = []
            for f in b["all_features"]:
                if f not in FEATS:
                    rows.append({"Feature": f, "Type": f"dropped - {b['schema']['dropped'][f]}"})
                    continue
                s = c[f]
                rows.append({"Feature": f, "Type": "numeric" if f in NUM else "categorical", "Missing": s.isna().mean(),
                             "Unique": s.nunique(), "Coerced text": int((b["data"][f].notna() & s.isna()).sum()) if f in NUM else None,
                             "Outliers (IQR)": iqr_outliers(s.dropna()) if f in NUM else None,
                             "Summary": (f"median {fmt(s.median())} · range [{fmt(s.min())}, {fmt(s.max())}] · skew {s.skew():.2f}" if f in NUM
                                         else ", ".join(f"{v} {p:.0%}" for v, p in s.value_counts(normalize=True).head(4).items()))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=320, column_config={
                "Missing": st.column_config.ProgressColumn("Missing", min_value=0, max_value=1, format="percent"),
                "Summary": st.column_config.TextColumn("Summary", width="large")})

        st.markdown("#### Feature explorer")
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        f = fc1.selectbox("Feature", FEATS, key="eda_feature")
        if f in NUM:
            norm = fc2.toggle("Normalise per class", True, help="Compare shapes even when classes differ in size")
            logx = fc3.toggle("Log x-axis", False)
            fig = px.histogram(c, x=f, color=LABEL, barmode="overlay", nbins=60, opacity=0.55, marginal="box",
                               histnorm="probability density" if norm else None, log_x=logx,
                               color_discrete_map=CMAP, category_orders={LABEL: CLASSES})
        else:
            ct = pd.crosstab(c[f].fillna("(missing)"), y)
            ct.index = [f"{v} (n={n:,})" for v, n in zip(ct.index, ct.sum(1))]
            long = ct.div(ct.sum(1), axis=0).rename_axis(f).reset_index().melt(id_vars=f, var_name=LABEL, value_name="share")
            fig = px.bar(long, y=f, x="share", color=LABEL, orientation="h", color_discrete_map=CMAP, text_auto=".0%",
                         category_orders={LABEL: CLASSES}, labels={"share": f"share of rows within each {f} level"})
            fig.update_xaxes(tickformat=".0%")
        show(fig, 420, key="eda_hist")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("#### Missing values")
            mv = c[FEATS].isna().mean().sort_values()
            fig = px.bar(x=mv.values, y=mv.index, orientation="h", text=[f"{v:.1%}" for v in mv.values], labels={"x": "share missing", "y": ""})
            fig.update_traces(marker_color=PALETTE[0], textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickformat=".0%")
            show(fig, 300)
        with m2:
            st.markdown(f"#### Is missingness informative? (`{FOCUS}` rate)")
            if len(miss):
                rr = pd.DataFrame([{"Feature": f, "Row": state, "rate": (y[mask] == FOCUS).mean(), "n": int(mask.sum())}
                                   for f in miss.index for state, mask in (("value missing", c[f].isna()), ("value present", c[f].notna()))])
                fig = px.bar(rr, x="Feature", y="rate", color="Row", barmode="group", text=[f"{v:.0%}" for v in rr.rate], hover_data=["n"],
                             color_discrete_map={"value missing": PALETTE[1], "value present": MUTED}, labels={"rate": f"share {FOCUS}"})
                fig.update_yaxes(tickformat=".0%")
                show(fig, 300)
            else:
                st.success("No missing values in the training data.")

        o1, o2 = st.columns(2)
        with o1:
            st.markdown("#### Outliers (robust z = (x - median) / IQR)")
            if NUM:
                rz = pd.concat([((c[f] - c[f].median()) / max(c[f].quantile(0.75) - c[f].quantile(0.25), 1e-9)).rename(f) for f in NUM], axis=1)
                fig = px.box(rz.melt(var_name="feature", value_name="robust z").dropna(), x="feature", y="robust z", points="outliers")
                fig.update_traces(marker_color=PALETTE[0], line_color=PALETTE[0], marker_size=4)
                show(fig, 360)
        with o2:
            st.markdown("#### Signal strength vs the label (mutual information)")
            fig = px.bar(x=mi.values, y=mi.index, orientation="h", labels={"x": "mutual information (nats)", "y": ""}, text=[f"{v:.3f}" for v in mi.values])
            fig.update_traces(marker_color=PALETTE[0], textposition="outside", cliponaxis=False)
            fig.update_yaxes(autorange="reversed")
            show(fig, 360)

        if len(NUM) >= 2:
            k1, k2 = st.columns(2)
            with k1:
                st.markdown("#### Correlation between numeric features")
                method = st.radio("Method", ["spearman", "pearson"], horizontal=True, key="corr_method",
                                  help="Spearman is rank-based: robust to the skew and outliers seen above")
                fig = px.imshow(c[NUM].corr(method), zmin=-1, zmax=1, color_continuous_scale=DIVERGING, text_auto=".2f", aspect="auto")
                show(fig, 380)
            with k2:
                st.markdown("#### Pairwise explorer")
                p1, p2 = st.columns(2)
                fx = p1.selectbox("x", NUM, index=0, key="pair_x")
                fy = p2.selectbox("y", NUM, index=min(1, len(NUM) - 1), key="pair_y")
                smp = c.sample(min(3000, len(c)), random_state=0)
                fig = px.scatter(smp, x=fx, y=fy, color=LABEL, opacity=0.55, color_discrete_map=CMAP, category_orders={LABEL: CLASSES})
                fig.update_traces(marker_size=6)
                if st.toggle("Zoom to 1st-99th percentile", True, key="pair_zoom"):
                    fig.update_xaxes(range=c[fx].quantile([0.01, 0.99]).tolist())
                    fig.update_yaxes(range=c[fy].quantile([0.01, 0.99]).tolist())
                show(fig, 300)

    # ------------------------------------------------------------------ preprocessing
    with prep:
        st.markdown("Every step below lives inside **one scikit-learn `Pipeline` fitted on the training split only**, "
                    "so the identical transformations run automatically on any file uploaded in Tab 2 - no manual steps, no leakage.")
        st.markdown("#### 1 · Cleaning log")
        st.dataframe(b["log"], hide_index=True, column_config={"Why": st.column_config.TextColumn("Why", width="large")})

        enc = b["prep"].named_steps["encode"]
        flags = [NUM[i] for i in enc.named_transformers_["missing_flag"].features_] if NUM else []
        dot = ['digraph{rankdir=LR; bgcolor="transparent"; node[shape=box style="rounded,filled" fillcolor="#f0efec" color="#c3c2b7" '
               'fontname="Helvetica" fontsize=11]; edge[color="#898781" fontname="Helvetica" fontsize=10];',
               'raw[label="Uploaded rows\\n(columns A-G)"]; clean[label="Cleaner\\n- tolerant number parsing\\n- trim + case-fold text\\n'
               '- placeholders (?, N/A...) -> missing"]; raw->clean;']
        if NUM:
            dot.append(f'num[label="Numeric: {", ".join(NUM)}\\n1. winsorise to 1st-99th pct\\n2. median imputation\\n'
                       f'3. standardise\\n4. Yeo-Johnson de-skew"]; clean->num->model;')
        if flags:
            dot.append(f'flag[label="Missing-value flags\\n{", ".join(flags)}"]; clean->flag->model;')
        if CAT:
            dot.append(f'cat[label="Categorical: {", ".join(CAT)}\\n1. missing -> own level\\n2. one-hot encode\\n'
                       f'(<1% or unseen -> infrequent)"]; clean->cat->model;')
        dot.append(f'model[label="{b["model_name"]}" fillcolor="#cde2fb"]; out[label="Class +\\nprobabilities"]; model->out;}}')
        st.markdown("#### 2 · Transformation pipeline")
        st.graphviz_chart("".join(dot), width="stretch")

        st.markdown("#### 3 · Parameters learned from the training rows")
        p1, p2 = st.columns([3, 2])
        if NUM:
            nump = enc.named_transformers_["num"]
            p1.dataframe(pd.DataFrame({
                "Feature": NUM, "Clip low (1%)": nump["clip_outliers"].lo_, "Clip high (99%)": nump["clip_outliers"].hi_,
                "Imputed median": nump["impute_median"].statistics_, "Yeo-Johnson λ": nump["yeo_johnson"].lambdas_,
                "Missing flag": ["yes" if f in flags else "-" for f in NUM]}).round(3), hide_index=True)
        if CAT:
            ohe = enc.named_transformers_["cat"]["one_hot"]
            infreq = ohe.infrequent_categories_ if getattr(ohe, "infrequent_categories_", None) is not None else [None] * len(CAT)
            p2.dataframe(pd.DataFrame({"Feature": CAT, "Levels kept": [", ".join(map(str, cats)) for cats in ohe.categories_],
                                       "Grouped as infrequent": [", ".join(map(str, i)) if i is not None else "-" for i in infreq]}),
                         hide_index=True)

        st.markdown("#### 4 · Before -> after")
        Z = transformed(b, KEY)
        f = st.selectbox("Feature", FEATS, key="prep_feature")
        tr = c[c.split == "train"]
        if f in NUM:
            fig = make_subplots(1, 2, subplot_titles=(f"{f} - raw (after parsing)", f"{f} - model-ready"))
            fig.add_histogram(x=tr[f], nbinsx=60, marker_color=MUTED, name="before", row=1, col=1)
            fig.add_histogram(x=Z[f], nbinsx=60, marker_color=PALETTE[0], name="after", row=1, col=2)
            show(fig, 320, showlegend=False)
            stats = pd.DataFrame({"Feature": NUM, "Missing before": [int(tr[g].isna().sum()) for g in NUM],
                                  "Skew before": [tr[g].skew() for g in NUM], "Skew after": [Z[g].skew() for g in NUM],
                                  "Mean before": [tr[g].mean() for g in NUM], "Std before": [tr[g].std() for g in NUM],
                                  "Mean after": [Z[g].mean() for g in NUM], "Std after": [Z[g].std() for g in NUM]})
            st.dataframe(stats.round(3), hide_index=True)
        else:
            raw = b["data"].loc[tr.index, f].astype(object).where(b["data"].loc[tr.index, f].notna(), "(blank)").astype(str)
            cols = [z for z in Z.columns if z.startswith(f"{f}_")]
            fig = make_subplots(1, 2, subplot_titles=(f"{f} - raw spellings ({raw.nunique()})", f"{f} - one-hot columns ({len(cols)})"))
            rv = raw.value_counts()
            fig.add_bar(x=rv.values, y=[repr(v) for v in rv.index], orientation="h", marker_color=MUTED, row=1, col=1)
            fig.add_bar(x=Z[cols].sum().values, y=cols, orientation="h", marker_color=PALETTE[0], row=1, col=2)
            show(fig, 360, showlegend=False)

        st.markdown("#### 5 · Sample rows: raw -> model-ready matrix")
        s1, s2 = st.columns(2)
        s1.dataframe(b["data"].loc[Z.index[:8], b["all_features"]], height=320)
        s2.dataframe(Z.head(8).round(3), height=320)

    # ------------------------------------------------------------------ split
    with split:
        n_tr, n_te, k = int((c.split == "train").sum()), int((c.split == "test").sum()), b["cv_folds"]
        st.markdown(f"""
- **10 % stratified test set ({n_te:,} rows)** is carved out first (fixed seed {pl.SEED}) and **never used for tuning or model
  choice**. It is scored exactly once, and it is the file uploaded in Tab 2.
- **90 % training set ({n_tr:,} rows)** feeds **{k}-fold stratified cross-validation** for both hyper-parameter search and
  model selection. Each fold re-fits the whole preprocessing pipeline on its own training part, so no statistic (median,
  clip bound, category list) leaks from validation rows.
- **Why not one fixed validation split?** With ~{n_tr:,} rows a single 10-20 % validation slice gives a noisy estimate and
  wastes data; {k}-fold CV validates on every training row once and reports a mean ± std, so differences between models
  can be judged against their noise.
- **Stratification** keeps class proportions identical across splits (table below). **De-duplication before the split**
  stops identical rows sitting on both sides, which would inflate the test score.
- **Final model** = winning configuration re-fit on all {n_tr:,} training rows, then evaluated once on the test set.""")
        st.graphviz_chart(
            'digraph{rankdir=LR; bgcolor="transparent"; node[shape=box style="rounded,filled" fillcolor="#f0efec" color="#c3c2b7" '
            'fontname="Helvetica" fontsize=11]; edge[color="#898781" fontname="Helvetica" fontsize=10];'
            f'all[label="{len(c):,} labeled rows\\n(after cleaning)"]; train[label="Training set\\n{n_tr:,} rows (90%)"]; '
            f'test[label="Test set\\n{n_te:,} rows (10%)\\nlocked away" fillcolor="#fde3d8"]; '
            f'cv[label="{k}-fold stratified CV\\nrandomised hyper-parameter search\\n+ model comparison"]; '
            f'final[label="Refit winner on\\nall {n_tr:,} training rows" fillcolor="#cde2fb"]; ev[label="Final test metrics\\n+ Tab 2 hold-out file"]; '
            'all->train[label="stratified"]; all->test[label="stratified, seed 42"]; train->cv; cv->final[label="best config"]; '
            'final->ev; test->ev[label="used once"];}', width="stretch")
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"#### {k}-fold cross-validation layout")
            z = np.eye(k)
            fig = go.Figure(go.Heatmap(z=z, x=[f"part {j + 1}" for j in range(k)], y=[f"fold {i + 1}" for i in range(k)],
                                       colorscale=[[0, "#cde2fb"], [1, PALETTE[1]]], showscale=False, xgap=3, ygap=3,
                                       text=np.where(z == 1, "validate", "train"), texttemplate="%{text}"))
            fig.update_yaxes(autorange="reversed")
            show(fig, 300)
        with f2:
            st.markdown("#### Class mix per split (stratification check)")
            mix = pd.crosstab(c.split, y, normalize="index").reindex(columns=CLASSES)
            counts = c.split.value_counts()
            mix.insert(0, "rows", counts.reindex(mix.index))
            st.dataframe(mix.style.format({cl: "{:.1%}" for cl in CLASSES} | {"rows": "{:,}"}))
            st.caption("Identical class shares in train and test = the test score is representative of the training distribution.")

    # ------------------------------------------------------------------ training & evaluation
    with model_tab:
        lb, t = b["leaderboard"], b["test"]
        st.markdown(f"Six model families compared, from simple to flexible: linear (**logistic regression**), distance-based "
                    f"(**k-NN**), bagged trees (**random forest, extra trees**), boosted trees (**HistGradientBoosting**), a "
                    f"**neural net (MLP)**, plus a **soft-voting ensemble** of the top three. Each is tuned by randomised search over "
                    f"its key hyper-parameters, scored by {b['cv_folds']}-fold CV accuracy. The winner is the highest mean CV "
                    f"accuracy: **{b['model_name']}**.")
        st.markdown("#### Leaderboard (cross-validation on the training set)")
        fig = go.Figure()
        sel = lb.Model == b["model_name"]
        fig.add_scatter(x=lb["CV accuracy"], y=lb.Model, mode="markers", name="CV accuracy (± 1 std)",
                        error_x=dict(type="data", array=lb["CV std"], color=MUTED, thickness=1.5),
                        marker=dict(size=12, color=np.where(sel, PALETTE[0], MUTED)))
        fig.add_scatter(x=lb["Train accuracy"], y=lb.Model, mode="markers", name="train accuracy (overfitting gap)",
                        marker=dict(size=9, symbol="diamond-open", color=PALETTE[1]))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".0%", title="accuracy")
        show(fig, 380)
        st.dataframe(lb, hide_index=True, column_config={
            **{m: st.column_config.NumberColumn(m, format="percent") for m in ("CV accuracy", "Train accuracy", "CV balanced acc.", "CV macro-F1")},
            "CV std": st.column_config.NumberColumn("CV std", format="%.4f"),
            "Fit time (s)": st.column_config.NumberColumn("Fit time (s)", format="%.2f"),
            "Best params": st.column_config.TextColumn("Best params", width="large")})

        st.markdown("#### Hyper-parameter search explorer")
        mname = st.selectbox("Model", list(b["searches"]), index=list(b["searches"]).index(b["model_name"]) if b["model_name"] in b["searches"] else 0)
        h = b["searches"][mname]
        dims = []
        for col in h.columns:
            v = h[col]
            if col != "CV accuracy" and pd.api.types.is_numeric_dtype(v) and v.notna().all() and v.min() > 0 and v.max() / v.min() > 100:
                dims.append(dict(label=f"log10({col})", values=np.log10(v.astype(float))))
            elif pd.api.types.is_numeric_dtype(v) and v.notna().all():
                dims.append(dict(label=col, values=v))
            else:
                codes, uniq = pd.factorize(v.astype(str))
                dims.append(dict(label=col, values=codes, tickvals=list(range(len(uniq))), ticktext=list(uniq)))
        fig = go.Figure(go.Parcoords(line=dict(color=h["CV accuracy"], colorscale=[[0, BLUES[1]], [1, BLUES[-1]]], showscale=True,
                                               colorbar=dict(title="CV acc.")), dimensions=dims))
        show(fig, 380, margin=dict(l=70, r=70, t=60, b=20))
        st.caption("Each line is one configuration tried; darker = higher CV accuracy. Drag along an axis to filter.")

        st.markdown("#### Selected model on the untouched 10 % test set")
        cv_acc = float(lb.loc[sel, "CV accuracy"].iloc[0])
        k = st.columns(6)
        k[0].metric("Test accuracy", f"{t['accuracy']:.1%}", f"{100 * (t['accuracy'] - t['baseline_accuracy']):+.1f} pts vs baseline")
        k[1].metric("CV accuracy", f"{cv_acc:.1%}", f"{100 * (t['accuracy'] - cv_acc):+.1f} pts test - CV", delta_color="off")
        k[2].metric("Balanced accuracy", f"{t['balanced_accuracy']:.1%}")
        k[3].metric("Macro F1", f"{t['f1_macro']:.3f}")
        k[4].metric("ROC-AUC", f"{t['roc_auc']:.3f}", help="One-vs-rest macro average for >2 classes")
        k[5].metric("Log-loss", f"{t['log_loss']:.3f}", help="Quality of the probabilities (lower is better)")

        e1, e2 = st.columns(2)
        with e1:
            norm = st.toggle("Normalise by actual class", False, key="cm_norm")
            cm = t["confusion"].astype(float)
            z = cm / cm.sum(1, keepdims=True) if norm else cm
            fig = px.imshow(z, x=CLASSES, y=CLASSES, text_auto=".1%" if norm else ".0f", color_continuous_scale=BLUES[1:],
                            labels=dict(x="Predicted", y="Actual", color="share" if norm else "rows"), aspect="auto")
            fig.update_xaxes(type="category")
            fig.update_yaxes(type="category")
            show(fig, 360, title="Confusion matrix")
        with e2:
            rep = pd.DataFrame(t["report"]).T
            rep.loc["accuracy", ["precision", "recall", "support"]] = [np.nan, np.nan, rep.loc["macro avg", "support"]]
            rep["support"] = rep["support"].astype(int)
            st.markdown("**Classification report**")
            st.dataframe(rep.round(3), height=250)
            st.caption("Precision: of rows predicted as a class, share correct. Recall: of rows truly in a class, share found.")

        r1, r2 = st.columns(2)
        with r1:
            fig = go.Figure()
            for cl, (fpr, tpr) in t["roc"].items():
                if len(CLASSES) == 2 and cl != CLASSES[1]:
                    continue
                fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{cl} vs rest", line=dict(width=2, color=CMAP.get(cl, PALETTE[0])))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(width=1, color=MUTED))
            fig.update_xaxes(title="false-positive rate")
            fig.update_yaxes(title="true-positive rate")
            show(fig, 360, title=f"ROC curve (AUC {t['roc_auc']:.3f})")
        with r2:
            conf = t["proba"].max(1)
            right = t["y_pred"] == t["y_true"]
            edges = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
            bins = pd.cut(conf, edges, right=False)
            rel = pd.DataFrame({"bin": bins, "right": right, "conf": conf}).groupby("bin", observed=True).agg(
                accuracy=("right", "mean"), confidence=("conf", "mean"), n=("right", "size")).reset_index()
            rel["bin"] = [f"{iv.left:.0%}-{min(iv.right, 1):.0%}" for iv in rel.bin]
            fig = go.Figure()
            fig.add_bar(x=rel.bin, y=rel.accuracy, name="observed accuracy", marker_color=PALETTE[0],
                        text=[f"n={n}" for n in rel.n], textposition="inside", insidetextanchor="start", textfont_color="white")
            fig.add_scatter(x=rel.bin, y=rel.confidence, mode="lines+markers", name="mean stated confidence", line=dict(color=PALETTE[1], width=2))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
            fig.update_xaxes(title="model confidence bucket")
            show(fig, 360, title="When the model is confident, is it right?")

        i1, i2 = st.columns(2)
        with i1:
            imp = b["importance"].sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="Std", labels={"Importance": "accuracy drop when shuffled"})
            fig.update_traces(marker_color=PALETTE[0])
            fig.update_xaxes(tickformat=".1%")
            show(fig, 360, title="Permutation importance (test set)")
        with i2:
            fp = st.selectbox("Partial dependence for", FEATS, index=FEATS.index(b["importance"].Feature.iloc[0]), key="pdp_feature")
            grid, P = partial_dependence(b, KEY, fp)
            fig = go.Figure()
            for j, cl in enumerate(CLASSES):
                if len(CLASSES) == 2 and j == 0:
                    continue
                if fp in NUM:
                    fig.add_scatter(x=grid, y=P[:, j], mode="lines+markers", name=f"P({cl})", line=dict(color=CMAP[cl], width=2), marker_size=5)
                else:
                    fig.add_bar(x=[str(g) for g in grid], y=P[:, j], name=f"P({cl})", marker_color=CMAP[cl])
            fig.update_xaxes(title=fp)
            fig.update_yaxes(title="average predicted probability", tickformat=".0%")
            show(fig, 300, title=f"How {fp} moves the prediction (all else as observed)", showlegend=True)


# ============================================================================= TAB 2
def waterfall(ctb):
    rows = ctb["contributions"]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(rows) + ["total"],
        y=["baseline (avg row)"] + [f"{r['feature']} = {r['value']}" for r in rows] + [f"this row"],
        x=[ctb["baseline_pct"]] + [r["effect_pts"] for r in rows] + [0],
        text=[f"{ctb['baseline_pct']:.1f}%"] + [f"{r['effect_pts']:+.1f}" for r in rows] + [f"{ctb['final_pct']:.1f}%"],
        textposition="outside", cliponaxis=False,
        increasing=dict(marker=dict(color=PALETTE[0])), decreasing=dict(marker=dict(color=PALETTE[1])),
        totals=dict(marker=dict(color=MUTED)), connector=dict(line=dict(color=GRID, width=1))))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"P({ctb['explaining_class']}) in %")
    return fig


@st.fragment
def predict_tab(llm):
    st.markdown(f"Upload rows with columns **{', '.join(b['all_features'])}**. The saved pipeline cleans, encodes and scores them "
                f"immediately. Columns are matched case-insensitively, extra columns are ignored, and if **{LABEL}** is present "
                "the predictions are scored against it.")
    u1, u2 = st.columns([3, 2])
    up = u1.file_uploader("Hold-out / new data (CSV or Excel)", type=FILE_TYPES, key="pred_upload")
    with u2:
        d1, d2 = st.columns(2)
        hold, hold_l = ART / "holdout_test.csv", ART / "holdout_test_labeled.csv"
        if hold.exists():
            d1.download_button("⬇ Hold-out set (A-G)", hold.read_bytes(), hold.name, "text/csv", on_click="ignore", width="stretch")
        if hold_l.exists():
            d2.download_button("⬇ Hold-out + labels", hold_l.read_bytes(), hold_l.name, "text/csv", on_click="ignore", width="stretch")
            if st.button("▶ Score the saved 10 % hold-out set", width="stretch"):
                st.session_state.pred_src = (hold_l.name, hold_l.read_bytes())
    if up is not None and st.session_state.get("pred_file_id") != up.file_id:
        st.session_state.pred_file_id = up.file_id
        st.session_state.pred_src = (up.name, up.getvalue())
    if "pred_src" not in st.session_state:
        st.info("Upload a file (or use the saved hold-out set) to run the pipeline.")
        return
    name, data = st.session_state.pred_src
    src = hashlib.md5(data).hexdigest()[:10]
    try:
        X, y, issues, out, proba, secs = process(b, KEY, name, data)
    except Exception as e:
        st.error(f"Could not process **{name}**: {e}")
        return

    st.success(f"**{name}** - {len(X):,} rows cleaned, encoded and scored by the trained pipeline ({b['model_name']}) "
               f"in {1000 * secs:,.0f} ms.", icon="✅")
    warns = [m for lvl, m in issues if lvl == "warning"]
    with st.expander(f"Data checks: {len(warns)} warning(s), {len(issues) - len(warns)} note(s)", expanded=bool(warns)):
        for lvl, m in issues:
            st.markdown(f"{'⚠️' if lvl == 'warning' else 'ℹ️'} {m}")
        if not issues:
            st.markdown("✅ Schema matched; every value parsed cleanly.")

    view = pd.concat([out, X], axis=1)
    if y is not None:
        ok = y.notna()
        hit = out.Prediction.astype(str) == y.astype(str)
        k = st.columns(4)
        k[0].metric("Accuracy vs uploaded labels", f"{hit[ok].mean():.1%}")
        k[1].metric("Balanced accuracy", f"{balanced_accuracy_score(y[ok], out.Prediction[ok]):.1%}")
        k[2].metric("Misclassified rows", f"{int((~hit & ok).sum()):,}")
        k[3].metric("Rows scored", f"{int(ok.sum()):,}")
        view.insert(2, "Actual", y)
        view.insert(3, "Correct", np.where(ok, np.where(hit, "✅", "❌"), ""))

    options = ["All rows", "Low confidence (< 60 %)"] + (["Misclassified"] if y is not None else [])
    shown = st.radio("Show", options, horizontal=True, key="pred_filter")
    mask = (view.Confidence < 0.6) if shown.startswith("Low") else (view.get("Correct") == "❌") if shown == "Misclassified" else view.Confidence.notna()
    view.insert(0, "Explain", False)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        ed_key = f"ed_{src}_{options.index(shown)}"
        edited = st.data_editor(
            view[mask], key=ed_key, height=35 * 21 + 3, disabled=[col for col in view.columns if col != "Explain"],
            column_config={"Explain": st.column_config.CheckboxColumn("💬 Explain", help="Tick to ask the agent about this row", width=110),
                           "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent")})
        st.caption(f"{int(mask.sum()):,} rows - scroll inside the table; tick **💬 Explain** on any row for an on-demand explanation.")
        checked = set(edited.index[edited.Explain])
        new = checked - st.session_state.get(ed_key + "_prev", set())
        st.session_state[ed_key + "_prev"] = checked
        if new:
            st.session_state.explain_row = (src, max(new))
    with right:
        pick = st.session_state.get("explain_row")
        if not pick or pick[0] != src or pick[1] not in X.index:
            with st.container(border=True):
                st.markdown("#### 💬 Prediction explainer")
                st.markdown("Tick **Explain** on a row. The agent then:\n1. computes exact **Shapley contributions** of A-G for that row,\n"
                            "2. compares the row's values with the **training data**,\n3. searches **what-if** changes that would flip the outcome,\n"
                            "4. writes a short, plain-language explanation.\n\nNothing is pre-computed: it runs only when asked.")
        else:
            r = pick[1]
            cache = st.session_state.setdefault("explanations", {})
            ck = (src, r, f"{llm['provider']}|{llm['model']}" if llm else "offline")
            if ck not in cache:
                with st.spinner(f"Agent investigating row {r}..."):
                    cache[ck] = agent.explain(b, X.loc[[r]], llm)
            e = cache[ck]
            with st.container(border=True):
                st.markdown(f"#### Row {r} -> **{out.at[r, 'Prediction']}** · {out.at[r, 'Confidence']:.0%} confidence")
                st.caption(f"Engine: {e['engine']}")
                st.markdown(e["text"])
                st.plotly_chart(waterfall(e["contrib"]).update_layout(height=300, margin=dict(l=8, r=40, t=30, b=8),
                                                                      title="Shapley contributions (probability pts)"),
                                key=f"wf_{src}_{r}", config={"displaylogo": False})
                with st.expander(f"Agent trace - {len(e['trace'])} tool call(s)"):
                    for step in e["trace"]:
                        st.markdown(f"**`{step['tool']}`** {json.dumps(step['args']) if step['args'] else ''}")
                        st.json(step["result"], expanded=False)
                if st.button("↻ Re-run explanation", key=f"rerun_{src}_{r}"):
                    cache.pop(ck, None)
                    st.rerun(scope="fragment")

    g1, g2, g3 = st.columns([2, 2, 1])
    with g1:
        vc = out.Prediction.astype(str).value_counts().reindex(CLASSES, fill_value=0)
        fig = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_map=CMAP, text=[f"{v:,}" for v in vc.values], labels={"x": "predicted", "y": "rows"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(type="category")
        show(fig, 260, key=f"pd_{src}", title="Predicted class mix", showlegend=False)
    with g2:
        fig = px.histogram(out, x="Confidence", nbins=20, labels={"Confidence": "confidence"})
        fig.update_traces(marker_color=PALETTE[0])
        fig.update_xaxes(tickformat=".0%")
        show(fig, 260, key=f"cf_{src}", title="Confidence distribution")
    with g3:
        probs = pd.DataFrame(proba, columns=[f"P({cl})" for cl in CLASSES], index=X.index)
        csv = pd.concat([X, out, probs] + ([y.rename(f"{LABEL} (actual)")] if y is not None else []), axis=1).to_csv(index_label="Row")
        st.download_button("⬇ Predictions CSV", csv, f"predictions_{name.rsplit('.', 1)[0]}.csv", "text/csv",
                           on_click="ignore", type="primary", width="stretch")


with tab2:
    predict_tab(llm)
