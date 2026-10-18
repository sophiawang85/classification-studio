"""On-demand prediction explainer (agentic).

The agent explains ONE row. Its tools are plain Python over the trained pipeline
(exact Shapley values, training-data context, what-if re-scoring), so every
number it quotes is computed, never generated. An LLM - any OpenAI-compatible
endpoint: local Ollama (open-source, no key), OpenAI, Anthropic - decides which
tools to call and writes the narrative. With no LLM available the same tools
feed a deterministic template, so "Explain" always works.
"""
import json
import os
import urllib.request
from math import factorial

import numpy as np
import pandas as pd

from pipeline import Cleaner

PROVIDERS = {
    "Ollama (local, open-source)": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "env": None},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4.1-mini", "env": "OPENAI_API_KEY"},
    "Anthropic (Claude)": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-haiku-5-5", "env": "ANTHROPIC_API_KEY"},
    "Custom (OpenAI-compatible)": {"base_url": "", "model": "", "env": "LLM_API_KEY"},
}
OLLAMA_PREFS = ("qwen3", "qwen2.5", "llama3.1", "llama3.2", "mistral")

SYSTEM = """You explain one prediction of a tabular classifier to a business user.
Investigate with the tools before answering: call get_feature_contributions first; use get_feature_context to judge
whether the driving values are high, low or unusual; use find_counterfactual (or what_if) to say what would change
the outcome. Only quote numbers returned by tools. Feature names are opaque codes - do not guess what they mean.
Reply in Markdown, under 170 words, with exactly these parts:
**Verdict** - one sentence: predicted class and confidence.
**Why** - 2-4 bullets, strongest first: `feature = value`, direction and size in probability points, and why
(compare with typical training values).
**What would change it** - one bullet.
**Caveats** - only if relevant (missing / unseen / out-of-range inputs, low confidence)."""

TOOL_SPECS = [{"type": "function", "function": f} for f in [
    {"name": "get_feature_contributions",
     "description": "Exact Shapley contribution of each feature to the probability of the predicted class, in probability points.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_feature_context",
     "description": "How this row's value compares with the training data (percentile, typical values per class, category mix, missing/unseen flags).",
     "parameters": {"type": "object", "properties": {"feature": {"type": "string", "description": "Feature name; omit for all features."}}}},
    {"name": "find_counterfactual",
     "description": "Search single-feature changes that would flip the prediction (closest changes first).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "what_if",
     "description": "Re-score this row after changing some feature values.",
     "parameters": {"type": "object", "properties": {"changes": {"type": "object", "description": 'e.g. {"A": 40, "C": "north"}'}},
                    "required": ["changes"]}},
    {"name": "get_model_card",
     "description": "Which model is used, its validated accuracy, and globally most important features.",
     "parameters": {"type": "object", "properties": {}}},
]]


def shapley(pipe, x: pd.DataFrame, background: pd.DataFrame, features: list):
    """Exact interventional Shapley values of every class probability for one row.

    v(S) = mean over background rows b of f(x on S, b elsewhere); all 2^p coalitions are scored in one batch.
    Returns base (K,), phi (p, K), fx (K,) with base + phi.sum(0) == fx.
    """
    # ponytail: exact enumeration is 2^p; fine for A-G (128 coalitions), switch to sampled permutations past ~14 features
    p = len(features)
    bg = background[features].iloc[: max(4, min(len(background), 2 ** 13 >> max(p - 7, 0)))].to_numpy(object)
    masks = ((np.arange(2 ** p)[:, None] >> np.arange(p)) & 1).astype(bool)        # coalition m holds feature j if bit j set
    grid = np.where(masks[:, None, :], x[features].to_numpy(object).reshape(1, 1, p), bg[None])
    v = pipe.predict_proba(pd.DataFrame(grid.reshape(-1, p), columns=features)).reshape(2 ** p, len(bg), -1).mean(1)
    w = np.array([factorial(s) * factorial(p - s - 1) / factorial(p) for s in np.minimum(masks.sum(1), p - 1)])
    phi = np.array([(w[S, None] * (v[S | (1 << j)] - v[S])).sum(0) for j in range(p) for S in [np.flatnonzero(~masks[:, j])]])
    return v[0], phi, v[-1]


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (int, float, np.number)):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def _nth(n):
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def make_tools(bundle: dict, x: pd.DataFrame):
    """Tools bound to one raw input row `x` (1-row DataFrame). Returns (tools, facts)."""
    pipe, feats, classes, stats = bundle["pipeline"], bundle["features"], [str(c) for c in bundle["classes"]], bundle["stats"]
    schema = bundle["schema"]
    clean = Cleaner(schema["numeric"], schema["categorical"]).transform(x).iloc[0]
    proba = pipe.predict_proba(x)[0]
    k = int(proba.argmax())
    pred = classes[k]
    base, phi, _ = shapley(pipe, x, bundle["background"], feats)
    by_name = {f.casefold(): f for f in feats}
    raw = x.iloc[0].to_dict()

    def pct(f, v):
        s = stats[f]["sorted"]
        return float(100 * np.searchsorted(s, v) / max(len(s), 1))

    def get_feature_contributions():
        rows = [{"feature": f, "value": _fmt(clean[f]), "effect_pts": round(100 * phi[j, k], 1)} for j, f in enumerate(feats)]
        return {"explaining_class": pred, "baseline_pct": round(100 * base[k], 1), "final_pct": round(100 * proba[k], 1),
                "contributions": sorted(rows, key=lambda r: -abs(r["effect_pts"])),
                "method": "exact Shapley values vs. a training background; effects add up from baseline to final probability"}

    def get_feature_context(feature: str = ""):
        out = {}
        for f in [by_name[feature.casefold()]] if feature and feature.casefold() in by_name else feats:
            s, v = stats[f], clean[f]
            if s["type"] == "numeric":
                out[f] = ({"value": "missing", "note": f"imputed with the training median {s['median']:.4g}"} if pd.isna(v) else
                          {"value": round(float(v), 4), "percentile_in_training": round(pct(f, v)),
                           "training_median": round(float(s["median"]), 4), "median_by_class": s["median_by_class"],
                           "outside_training_range": bool(v < s["min"] or v > s["max"])})
            else:
                key = "missing" if pd.isna(v) else v
                out[f] = {"value": key, "share_of_training_rows": s["share"].get(key, 0.0),
                          "class_mix_for_this_value": s["class_mix"].get(key, "unseen in training"),
                          "overall_class_mix": {str(c): round(p, 3) for c, p in bundle["class_mix"].items()}}
        return out

    def score(rows):
        return pipe.predict_proba(pd.DataFrame(rows, columns=list(x.columns)))

    def find_counterfactual():
        tries = []
        for f in feats:
            s = stats[f]
            vals = (np.unique(np.quantile(s["sorted"], np.linspace(0.02, 0.98, 25))) if s["type"] == "numeric" and len(s["sorted"])
                    else [c for c in s["share"] if c != "missing"][:15])
            tries += [(f, v) for v in vals if _fmt(v) != _fmt(clean[f])]
        if not tries:
            return {"flips": []}
        P = score([{**raw, f: v} for f, v in tries])
        flips = []
        for f in feats:
            idx = [i for i, (g, _) in enumerate(tries) if g == f and P[i].argmax() != k]
            if not idx:
                continue
            cur = pct(f, clean[f]) if stats[f]["type"] == "numeric" and pd.notna(clean[f]) else 50.0
            i = min(idx, key=lambda i: abs(pct(f, tries[i][1]) - cur) if stats[f]["type"] == "numeric" else -P[i].max())
            flips.append({"feature": f, "from": _fmt(clean[f]), "to": _fmt(tries[i][1]), "new_prediction": classes[P[i].argmax()],
                          "new_confidence": round(float(P[i].max()), 3),
                          "size_of_change": f"{abs(pct(f, tries[i][1]) - cur):.0f} percentile points" if stats[f]["type"] == "numeric" else "category swap"})
        if flips:
            return {"flips": flips[:4]}
        i = int(P[:, k].argmin())
        return {"flips": [], "note": "no single-feature change flips the prediction",
                "largest_drop": {"feature": tries[i][0], "to": _fmt(tries[i][1]), f"P({pred})": round(float(P[i, k]), 3)}}

    def what_if(changes: dict):
        known = {by_name[str(f).casefold()]: v for f, v in (changes or {}).items() if str(f).casefold() in by_name}
        p2 = score([{**raw, **known}])[0]
        return {"applied_changes": known, "ignored": [f for f in (changes or {}) if str(f).casefold() not in by_name],
                "new_prediction": classes[int(p2.argmax())], "new_probabilities": {c: round(float(q), 3) for c, q in zip(classes, p2)}}

    def get_model_card():
        lb = bundle["leaderboard"].set_index("Model")
        return {"model": bundle["model_name"], "classes": classes,
                "cv_accuracy": round(float(lb.loc[bundle["model_name"], "CV accuracy"]), 4),
                "test_accuracy": round(float(bundle["test"]["accuracy"]), 4),
                "top_features_globally": bundle["importance"]["Feature"].head(5).tolist()}

    tools = {f.__name__: f for f in (get_feature_contributions, get_feature_context, find_counterfactual, what_if, get_model_card)}
    facts = {"row": {f: _fmt(clean[f]) for f in feats}, "predicted_class": pred, "confidence": round(float(proba[k]), 3),
             "probabilities": {c: round(float(q), 3) for c, q in zip(classes, proba)}}
    return tools, facts


def _json(o):
    return o.item() if hasattr(o, "item") else str(o)


def run_agent(tools, facts, llm, client=None, max_steps=6):
    """Tool-calling loop: the LLM picks tools until it is ready to answer."""
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=llm["base_url"], api_key=llm.get("api_key") or "none", timeout=120, max_retries=1)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Explain this prediction:\n" + json.dumps(facts, default=_json)}]
    trace = []
    for step in range(max_steps):
        if step == max_steps - 1:
            msgs.append({"role": "user", "content": "Write the final answer now, without further tool calls."})
        msg = client.chat.completions.create(model=llm["model"], messages=msgs, tools=TOOL_SPECS).choices[0].message
        if not msg.tool_calls:
            if not trace and step == 0:  # answered without evidence: push it back to the tools once
                msgs += [{"role": "assistant", "content": msg.content or ""},
                         {"role": "user", "content": "Call get_feature_contributions first and base the answer on it."}]
                continue
            return (msg.content or "").strip(), trace
        if step == max_steps - 1:
            break
        msgs.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [{"id": t.id, "type": "function",
                                     "function": {"name": t.function.name, "arguments": t.function.arguments or "{}"}}
                                    for t in msg.tool_calls]})
        for t in msg.tool_calls:
            args = {}
            try:
                args = json.loads(t.function.arguments or "{}") or {}
                result = tools[t.function.name](**args)
            except Exception as e:  # bad tool name/arguments go back to the model, not to the user
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": t.function.name, "args": args, "result": result})
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(result, default=_json)})
    return "", trace


def offline_text(facts, contrib, ctx, cf) -> str:
    pred, conf = facts["predicted_class"], facts["confidence"]
    others = sorted(((p, c) for c, p in facts["probabilities"].items() if c != pred), reverse=True)
    lines = [f"**Verdict** - predicted **{pred}** with {conf:.0%} confidence"
             + (f" (next most likely: {others[0][1]} at {others[0][0]:.0%})." if others else "."), "", "**Why**"]
    for r in [r for r in contrib["contributions"] if abs(r["effect_pts"]) >= 0.5][:4] or contrib["contributions"][:1]:
        c = ctx[r["feature"]]
        if c["value"] == "missing":
            why = c.get("note", "missing in this row")
        elif "percentile_in_training" in c:
            why = f"{_nth(c['percentile_in_training'])} percentile of training data (median {_fmt(c['training_median'])})"
        else:
            mix = c["class_mix_for_this_value"]
            why = (f"category never seen in training" if isinstance(mix, str) else
                   f"{c['share_of_training_rows']:.0%} of training rows; {mix.get(pred, 0):.0%} of those are {pred}")
        verb = "raises" if r["effect_pts"] > 0 else "lowers"
        lines.append(f"- `{r['feature']} = {r['value']}` {verb} P({pred}) by {abs(r['effect_pts']):.1f} pts - {why}")
    lines += ["", "**What would change it**"]
    if cf["flips"]:
        f = cf["flips"][0]
        lines.append(f"- Changing `{f['feature']}` from {f['from']} to {f['to']} would flip it to **{f['new_prediction']}** "
                     f"({f['new_confidence']:.0%}).")
    else:
        lines.append("- No single-feature change flips this prediction - it is robust.")
    caveats = [f"`{k}` is missing (imputed)" for k, c in ctx.items() if c["value"] == "missing"]
    caveats += [f"`{k}` is outside the training range" for k, c in ctx.items() if c.get("outside_training_range")]
    caveats += [f"`{k}` has an unseen category" for k, c in ctx.items() if isinstance(c.get("class_mix_for_this_value"), str)]
    if conf < 0.6:
        caveats.append("low confidence - close to the decision boundary")
    if caveats:
        lines += ["", "**Caveats** - " + "; ".join(caveats) + "."]
    lines.append(f"\n_Baseline P({pred}) {contrib['baseline_pct']:.1f}% -> {contrib['final_pct']:.1f}% for this row._")
    return "\n".join(lines)


def explain(bundle: dict, x: pd.DataFrame, llm: dict | None = None) -> dict:
    tools, facts = make_tools(bundle, x)
    contrib = tools["get_feature_contributions"]()
    text, trace, engine = "", [], "Offline explainer (no LLM configured) - same tools, templated wording"
    if llm:
        try:
            text, trace = run_agent(tools, facts, llm)
            engine = f"{llm['provider']} - {llm['model']} (tool-calling agent)"
        except Exception as e:
            engine = f"Offline explainer - LLM call failed ({type(e).__name__}: {str(e)[:200]})"
    if not text:
        ctx, cf = tools["get_feature_context"](), tools["find_counterfactual"]()
        trace = [{"tool": n, "args": {}, "result": r} for n, r in
                 [("get_feature_contributions", contrib), ("get_feature_context", ctx), ("find_counterfactual", cf)]]
        text = offline_text(facts, contrib, ctx, cf)
    return {"text": text, "engine": engine, "trace": trace, "contrib": contrib, "facts": facts}


# ----------------------------------------------------------------------------- LLM discovery
def ollama_models(url="http://localhost:11434") -> list:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1) as r:
            return [m["name"] for m in json.load(r).get("models", []) if "embed" not in m["name"]]
    except Exception:
        return []


def auto_llm() -> dict | None:
    """Env key (OpenAI, then Anthropic) -> running Ollama -> None (offline explainer)."""
    for name in ("OpenAI", "Anthropic (Claude)"):
        p = PROVIDERS[name]
        if os.environ.get(p["env"]):
            return {"provider": name, "base_url": p["base_url"], "model": p["model"], "api_key": os.environ[p["env"]]}
    models = ollama_models()
    if models:
        pick = next((m for pref in OLLAMA_PREFS for m in models if m.startswith(pref)), models[0])
        return {"provider": "Ollama (local, open-source)", "base_url": PROVIDERS["Ollama (local, open-source)"]["base_url"],
                "model": pick, "api_key": "ollama"}
    return None
