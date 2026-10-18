"""Training + inference pipeline for the A-G -> H classifier.

Everything needed at prediction time (cleaning, outlier clipping, imputation,
scaling, encoding, model) lives in ONE sklearn Pipeline that is fitted on the
training split only. An uploaded file therefore goes through exactly the
transformations learned during training, with no manual steps.
"""
import io
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
FEATURES, LABEL = list("ABCDEFG"), "H"
SEED, TEST_SIZE = 42, 0.10
NULLS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "missing", "#n/a", "nat"}
SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}


# ----------------------------------------------------------------------------- reading & cleaning
def read_table(data: bytes, name: str) -> pd.DataFrame:
    """CSV/TSV/TXT (delimiter sniffed, utf-8 or latin-1) or Excel (first sheet)."""
    if not data:
        raise ValueError("The file is empty.")
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("The file has a header but no data rows.")
    return df


def to_numeric(s: pd.Series) -> pd.Series:
    """Tolerant number parsing: '1,234', '$5', ' 7 ', '12%' -> numbers; anything else -> NaN."""
    out = pd.to_numeric(s, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out = out.astype(float)
        out[bad] = pd.to_numeric(s[bad].astype(str).str.replace(r"[,\s$€£%]", "", regex=True), errors="coerce")
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def clean_cat(s: pd.Series) -> pd.Series:
    """Trim + case-fold categories, unify '1.0'/'1', map placeholders ('N/A', '?', ...) to NaN."""
    t = s.astype(str).str.strip().str.casefold().str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return t.where(~t.isin(NULLS)).astype(object)


def clean_label(s: pd.Series) -> pd.Series:
    """Labels become canonical strings (' 1.0 ' -> '1'); placeholders -> NaN."""
    t = s.astype(str).str.strip().str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return t.where(~t.str.casefold().isin(NULLS)).astype(object)


class Cleaner(BaseEstimator, TransformerMixin):
    """Coerce raw columns to the training schema. Stateless; absent columns become all-missing."""

    def __init__(self, numeric=(), categorical=()):
        self.numeric, self.categorical = numeric, categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        nan = pd.Series(np.nan, index=X.index, dtype=object)
        cols = {c: to_numeric(X[c] if c in X else nan) for c in self.numeric}
        cols |= {c: clean_cat(X[c] if c in X else nan) for c in self.categorical}
        return pd.DataFrame(cols, index=X.index)


class Clipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to its [q, 1-q] training quantiles (outlier treatment)."""

    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.nanquantile(X, self.q, axis=0), np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def infer_schema(X: pd.DataFrame) -> dict:
    """Numeric if >=90% of non-empty values parse as numbers, else categorical. Drops useless columns."""
    schema = {"numeric": [], "categorical": [], "dropped": {}}
    for c in X.columns:
        s = clean_cat(X[c]).dropna()
        num = to_numeric(X[c]).dropna()
        if s.nunique() <= 1:
            schema["dropped"][c] = "constant or empty"
        elif len(num) >= 0.9 * len(s):
            d = num.diff().dropna()
            if num.is_unique and len(d) and (d == 1).all():
                schema["dropped"][c] = "row index (1, 2, 3, ...)"
            else:
                schema["numeric"].append(c)
        elif s.nunique() > max(50, 0.9 * len(s)):
            schema["dropped"][c] = "identifier-like (unique text per row)"
        else:
            schema["categorical"].append(c)
    return schema


def make_preprocessor(schema: dict) -> Pipeline:
    num, cat = schema["numeric"], schema["categorical"]
    numeric = Pipeline([("clip_outliers", Clipper(0.01)),
                        ("impute_median", SimpleImputer(strategy="median")),
                        ("standardise", StandardScaler()),
                        ("yeo_johnson", PowerTransformer())])
    categorical = Pipeline([("impute_missing", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                                      max_categories=25, sparse_output=False))])
    encode = ColumnTransformer([("num", numeric, num),
                                ("missing_flag", MissingIndicator(error_on_new=False), num),
                                ("cat", categorical, cat)])
    return Pipeline([("clean", Cleaner(num, cat)), ("encode", encode)])


# ----------------------------------------------------------------------------- training
def candidates(fast=False) -> dict:
    """name -> (estimator, search space, random-search iterations)."""
    trees = {"max_depth": [None, 8, 12, 20], "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", 0.5, 0.8]}
    m = {
        "Logistic Regression": (LogisticRegression(max_iter=3000),
                                {"C": loguniform(1e-3, 1e2), "class_weight": [None, "balanced"]}, 12),
        "k-Nearest Neighbours": (KNeighborsClassifier(),
                                 {"n_neighbors": randint(5, 80), "weights": ["uniform", "distance"], "p": [1, 2]}, 12),
        "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=SEED), trees, 12),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=300, random_state=SEED), trees, 10),
        "Gradient Boosting (HistGB)": (HistGradientBoostingClassifier(random_state=SEED),
                                       {"learning_rate": loguniform(0.02, 0.3), "max_iter": randint(100, 600),
                                        "max_leaf_nodes": randint(8, 64), "min_samples_leaf": randint(10, 100),
                                        "l2_regularization": loguniform(1e-3, 10)}, 20),
        "Neural Net (MLP)": (MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED),
                             {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                              "alpha": loguniform(1e-5, 1e-1), "learning_rate_init": loguniform(1e-4, 1e-2)}, 10),
    }
    if fast:  # used by the smoke test
        return {k: (e, s, 2) for k, (e, s, _) in m.items() if k in ("Logistic Regression", "Gradient Boosting (HistGB)")}
    return m


def _row(name, get, fit_time, params):
    acc = get("test", "accuracy")
    return {"Model": name, "CV accuracy": acc.mean(), "CV std": acc.std(), "Train accuracy": get("train", "accuracy").mean(),
            "CV balanced acc.": get("test", "balanced_accuracy").mean(), "CV macro-F1": get("test", "f1_macro").mean(),
            "Fit time (s)": fit_time, "Best params": params}


def feature_stats(Xc: pd.DataFrame, y: pd.Series, schema: dict) -> dict:
    """Training-set reference values the explainer compares a row against."""
    stats = {}
    for c in schema["numeric"]:
        s = Xc[c]
        stats[c] = {"type": "numeric", "sorted": np.sort(s.dropna().to_numpy()), "median": s.median(),
                    "min": s.min(), "max": s.max(), "median_by_class": s.groupby(y).median().round(4).to_dict()}
    for c in schema["categorical"]:
        s = Xc[c].fillna("missing")
        stats[c] = {"type": "categorical", "share": s.value_counts(normalize=True).round(4).to_dict(),
                    "class_mix": pd.crosstab(s, y, normalize="index").round(3).to_dict("index")}
    return stats


def train(df: pd.DataFrame, source="uploaded file", out_dir=ART, fast=False, progress=lambda frac, msg: None) -> dict:
    t0 = time.time()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    names = {c.casefold(): c for c in df.columns}
    label = names.get(LABEL.casefold(), df.columns[-1])
    feats = ([names[f.casefold()] for f in FEATURES] if all(f.casefold() in names for f in FEATURES)
             else [c for c in df.columns if c != label])

    log = []

    def note(step, affected, why):
        log.append({"Step": step, "Affected": int(affected), "Rows remaining": len(df), "Why": why})

    note("Loaded", len(df), f"{df.shape[1]} columns from {source}; label = '{label}', features = {', '.join(feats)}")
    y = clean_label(df[label])
    n0 = len(df)
    df, y = df[y.notna()], y[y.notna()]
    note("Dropped rows with missing label", n0 - len(df), "a row without a label cannot be learned from")
    if y.nunique() > 50 or (y.nunique() > 20 and y.nunique() > 0.05 * len(y)):
        raise ValueError(f"Label '{label}' has {y.nunique()} distinct values - it looks continuous, not a class label.")

    schema = infer_schema(df[feats])
    used = schema["numeric"] + schema["categorical"]
    if not used:
        raise ValueError("No usable feature columns (all constant, empty or identifiers).")
    for c, why in schema["dropped"].items():
        note(f"Dropped column {c}", 0, why)
    Xc = Cleaner(schema["numeric"], schema["categorical"]).transform(df)
    for c in schema["numeric"]:
        bad = (df[c].notna() & Xc[c].isna()).sum()
        if bad:
            note(f"Coerced text in numeric column {c}", bad, "placeholders like '?' -> missing (then imputed); '1,234' -> 1234")
    for c in schema["categorical"]:
        raw, cleaned = df[c].dropna().astype(str).nunique(), Xc[c].nunique()
        if raw > cleaned:
            note(f"Merged spelling variants in {c}", raw - cleaned, f"{raw} raw spellings -> {cleaned} categories (trim + case-fold)")

    dup = pd.concat([Xc[used], y], axis=1).duplicated()
    df, y, Xc = df[~dup], y[~dup], Xc[~dup]
    note("Removed exact duplicate rows", dup.sum(), "keeps copies of one row from landing in both train and test")
    conflicts = Xc[used].duplicated(keep=False).sum()
    if conflicts:
        note("Rows sharing features but not label (kept)", conflicts, "irreducible label noise - caps achievable accuracy")
    counts = y.value_counts()
    rare = counts[counts < 3].index
    if len(rare):
        keep = ~y.isin(rare)
        df, y, Xc = df[keep], y[keep], Xc[keep]
        note(f"Dropped ultra-rare classes {list(rare)}", (~keep).sum(), "fewer than 3 examples cannot be split and validated")
    if y.nunique() < 2:
        raise ValueError("Need at least two classes in the label column.")
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows - too few to train and validate a model.")

    # ---- split: stratified 10% test set, untouched until the final evaluation
    X = df[feats]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    k = int(max(2, min(5, y_tr.value_counts().min())))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    # ---- model selection: randomized search (preprocessing inside every fold -> no leakage)
    cands = candidates(fast)
    steps = len(cands) + 3
    board, searches, fitted = [], {}, {}
    for i, (name, (est, space, n_iter)) in enumerate(cands.items()):
        progress(i / steps, f"Tuning {name} ({n_iter} configs x {k} folds)")
        pipe = Pipeline([("prep", make_preprocessor(schema)), ("model", est)])
        try:
            s = RandomizedSearchCV(pipe, {f"model__{p}": v for p, v in space.items()}, n_iter=n_iter, cv=cv,
                                   scoring=SCORING, refit="accuracy", return_train_score=True, n_jobs=-1,
                                   random_state=SEED).fit(X_tr, y_tr)
        except Exception as e:  # one failing model must not sink the run
            note(f"Skipped {name}", 0, f"failed to train: {e}")
            continue
        r, b = s.cv_results_, s.best_index_
        params = {p.replace("model__", ""): v for p, v in s.best_params_.items()}
        board.append(_row(name, lambda part, m: np.array([r[f"split{j}_{part}_{m}"][b] for j in range(k)]),
                          r["mean_fit_time"][b], ", ".join(f"{p}={v:.3g}" if isinstance(v, float) else f"{p}={v}"
                                                            for p, v in params.items())))
        hist = pd.DataFrame(r["params"]).rename(columns=lambda c: c.replace("model__", ""))
        hist["CV accuracy"] = r["mean_test_accuracy"]
        searches[name], fitted[name] = hist, s.best_estimator_
    if not fitted:
        raise ValueError("Every candidate model failed to train - check the data.")

    ranked = sorted(board, key=lambda row: -row["CV accuracy"])
    if len(ranked) >= 3:
        progress(len(cands) / steps, "Cross-validating a soft-voting ensemble of the top 3")
        top = [row["Model"] for row in ranked[:3]]
        vote = VotingClassifier([(n, fitted[n]) for n in top], voting="soft")
        res = cross_validate(vote, X_tr, y_tr, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=-1)
        board.append(_row("Soft-voting ensemble", lambda part, m: res[f"{part}_{m}"], res["fit_time"].mean(),
                          "average of probabilities from " + ", ".join(top)))
        fitted["Soft-voting ensemble"] = vote
    res = cross_validate(DummyClassifier(), X_tr, y_tr, cv=cv, scoring=SCORING, return_train_score=True)
    board.append(_row("Baseline (always majority class)", lambda part, m: res[f"{part}_{m}"], res["fit_time"].mean(), "-"))
    board = pd.DataFrame(board).sort_values("CV accuracy", ascending=False, ignore_index=True)
    best_name = board.loc[board.Model != "Baseline (always majority class)", "Model"].iloc[0]
    model = fitted[best_name]
    if best_name == "Soft-voting ensemble":
        model.fit(X_tr, y_tr)

    # ---- final, one-time evaluation on the untouched test set
    progress((len(cands) + 1) / steps, f"Evaluating {best_name} on the 10% test set")
    classes = list(model.classes_)
    proba = model.predict_proba(X_te)
    pred = np.asarray(classes)[proba.argmax(1)]
    try:
        auc = (roc_auc_score(y_te == classes[1], proba[:, 1]) if len(classes) == 2
               else roc_auc_score(y_te, proba, multi_class="ovr", labels=classes))
    except ValueError:
        auc = float("nan")
    curves = {}
    for j, c in enumerate(classes):
        if (y_te == c).any() and (y_te != c).any():
            fpr, tpr, _ = roc_curve(y_te == c, proba[:, j])
            curves[c] = (fpr, tpr)
    test = {"accuracy": accuracy_score(y_te, pred), "balanced_accuracy": balanced_accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"), "log_loss": log_loss(y_te, proba, labels=classes),
            "roc_auc": auc, "baseline_accuracy": float((y_te == y_tr.value_counts().idxmax()).mean()),
            "confusion": confusion_matrix(y_te, pred, labels=classes),
            "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
            "y_true": y_te.to_numpy(), "y_pred": pred, "proba": proba, "roc": curves}
    progress((len(cands) + 2) / steps, "Permutation importance on the test set")
    pi = permutation_importance(model, X_te[used], y_te, n_repeats=10, random_state=SEED, scoring="accuracy", n_jobs=-1)
    importance = pd.DataFrame({"Feature": used, "Importance": pi.importances_mean, "Std": pi.importances_std}
                              ).sort_values("Importance", ascending=False, ignore_index=True)

    # ---- persist: model bundle + the exact hold-out file for Tab 2
    data = df[feats].copy()
    data[label] = y
    data["split"] = np.where(data.index.isin(X_te.index), "test", "train")
    bundle = {
        "pipeline": model, "model_name": best_name, "features": used, "all_features": feats, "label": label,
        "schema": schema, "classes": classes, "data": data, "source": source,
        "trained_at": datetime.now().isoformat(timespec="seconds"), "log": pd.DataFrame(log),
        "leaderboard": board, "searches": searches, "test": test, "importance": importance, "cv_folds": k,
        "prep": make_preprocessor(schema).fit(X_tr), "background": X_tr[used].sample(min(64, len(X_tr)), random_state=SEED),
        "stats": feature_stats(Xc.loc[X_tr.index], y_tr, schema), "class_mix": y_tr.value_counts(normalize=True).to_dict(),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.loc[X_te.index, feats].to_csv(out_dir / "holdout_test.csv", index=False)
    df.loc[X_te.index, feats + [label]].to_csv(out_dir / "holdout_test_labeled.csv", index=False)
    bundle["train_seconds"] = time.time() - t0
    joblib.dump(bundle, out_dir / "model.joblib", compress=3)
    progress(1.0, f"Done in {bundle['train_seconds']:.0f}s - selected {best_name}")
    return bundle


# ----------------------------------------------------------------------------- inference
def align(df: pd.DataFrame, bundle: dict):
    """Map an uploaded table onto the training columns. Returns (X, y_true | None, issues)."""
    feats, used, label = bundle["all_features"], bundle["features"], bundle["label"]
    issues = []
    lookup = {str(c).strip().casefold(): c for c in df.columns}
    found = {f: lookup[f.casefold()] for f in feats if f.casefold() in lookup}
    if not found:
        if df.shape[1] < len(feats):
            raise ValueError(f"Expected columns {', '.join(feats)}; the file has {', '.join(map(str, df.columns))}.")
        if to_numeric(pd.Series(list(df.columns), dtype=object)).notna().any():  # header row is really data
            df = pd.concat([pd.DataFrame([list(df.columns)], columns=df.columns), df], ignore_index=True)
            issues.append(("warning", "No header row found - the first line was treated as data."))
        found = dict(zip(feats, df.columns[:len(feats)]))
        issues.append(("warning", f"Column names did not match {', '.join(feats)}; mapped the first {len(feats)} columns by position."))
    X = pd.DataFrame({f: df[found[f]] if f in found else np.nan for f in feats}, index=df.index)
    X.index = pd.RangeIndex(1, len(X) + 1, name="Row")
    missing = [f for f in used if f not in found]
    if missing:
        issues.append(("warning", f"Missing column(s) {', '.join(missing)} - treated as missing values and imputed; predictions are less reliable."))
    extra = [str(c) for c in df.columns if c not in found.values() and str(c).strip().casefold() != label.casefold()]
    if extra:
        issues.append(("info", f"Ignored extra column(s): {', '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}"))
    y = None
    if label.casefold() in lookup and lookup[label.casefold()] not in found.values():
        y = clean_label(df[lookup[label.casefold()]]).set_axis(X.index)
        issues.append(("info", f"Label column '{label}' found - predictions will be scored against it."))

    clean = Cleaner(bundle["schema"]["numeric"], bundle["schema"]["categorical"]).transform(X)
    for c in bundle["schema"]["numeric"]:
        if c in missing:
            continue
        bad = X[c].notna() & clean[c].isna()
        if bad.any():
            ex = ", ".join(map(repr, X.loc[bad, c].astype(str).unique()[:3]))
            issues.append(("warning", f"{c}: {bad.sum()} non-numeric value(s) ({ex}) treated as missing and imputed."))
        ref = bundle["stats"][c]
        out = (clean[c] < ref["min"]) | (clean[c] > ref["max"])
        if out.any():
            issues.append(("info", f"{c}: {out.sum()} value(s) outside the training range [{ref['min']:g}, {ref['max']:g}] - clipped to the learned bounds."))
    for c in bundle["schema"]["categorical"]:
        if c in missing:
            continue
        unseen = clean[c].dropna()[~clean[c].dropna().isin(bundle["stats"][c]["share"])]
        if len(unseen):
            issues.append(("warning", f"{c}: {len(unseen)} row(s) with categories never seen in training "
                                      f"({', '.join(map(repr, unseen.unique()[:3]))}) - encoded as 'unknown/infrequent'."))
    n_miss = int(clean[[c for c in used if c not in missing]].isna().sum().sum())
    if n_miss:
        issues.append(("info", f"{n_miss} missing cell(s) filled with training medians / a 'missing' category."))
    return X, y, issues


def predict(bundle: dict, X: pd.DataFrame):
    proba = bundle["pipeline"].predict_proba(X)
    classes = np.asarray(bundle["classes"])
    out = pd.DataFrame({"Prediction": classes[proba.argmax(1)], "Confidence": proba.max(1)}, index=X.index)
    return out, proba
