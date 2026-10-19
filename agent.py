"""On-demand prediction explainer (agentic).

The agent explains ONE row and answers follow-up questions about it. Its tools are plain Python over the trained
pipeline (exact Shapley values, training-data context, what-if re-scoring, counterfactual search), so every number it
quotes is computed, never generated. A local open-source LLM served by Ollama decides which tools to call and writes
the answer. If Ollama isn't reachable, the same tools feed a deterministic template, so "Explain" always works.

Backend settings (environment variables, never shown in the UI):
  OLLAMA_HOST     default http://localhost:11434
  OLLAMA_MODEL    default qwen2.5:7b if installed, else the best installed chat model
  OLLAMA_API_KEY  optional bearer token for a remote/hosted Ollama server
"""
import json
import os
import re
import urllib.request
from math import factorial

import numpy as np
import pandas as pd

from pipeline import Cleaner

DEFAULT_MODEL = "qwen2.5:7b"
MODEL_PREFS = ("qwen2.5", "qwen3", "llama3.1", "llama3.2", "mistral", "gpt-oss")

SYSTEM = """You explain one prediction of a tabular classifier to a business user. Tools have already gathered the
evidence; it is given to you as verified sentences. Write the explanation ONLY from those sentences: keep every number
and direction exactly as written, never compute or compare values yourself, and never mention a feature, value or
change that is not in the evidence. Feature names are opaque codes - do not guess what they mean.
Reply in Markdown, under 130 words, with exactly these three parts (caveats are added separately - do not write any):
**Verdict** - one sentence: predicted class and confidence.
**Why** - 2-4 bullets for the largest contributions, strongest first: the contribution sentence plus the matching
comparison with the training data.
**What would change it** - the first "Changing ..." sentence from the evidence, or say the prediction is robust."""

CHAT_SYSTEM = """You are the explanation agent of a tabular classifier, answering a business user's follow-up questions
about ONE prediction. Verified evidence about it is in the conversation. When a question needs numbers that are not
there yet, call a tool: what_if to re-score changed values, find_counterfactual for what would flip it,
get_feature_context to compare a value with the training data, get_model_card for questions about the model itself.
Tool results come back as verified sentences. Answer ONLY from verified sentences: keep their numbers and directions
exactly, never compute or compare values yourself, never invent features, values or changes. Feature names are opaque
codes - do not guess what they mean. If the question assumes something the evidence contradicts, say so and correct
it. Pass what_if only the changes the user asked about. Answer in at most 4 short sentences or bullets, in Markdown."""

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
        rows = [{"feature": f, "value": _fmt(clean[f]), "effect_pts": round(100 * phi[j, k], 1),
                 "summary": f"{f} = {_fmt(clean[f])} {'raises' if phi[j, k] > 0 else 'lowers'} P({pred}) by {abs(100 * phi[j, k]):.1f} points"}
                for j, f in enumerate(feats)]
        return {"explaining_class": pred, "baseline_pct": round(100 * base[k], 1), "final_pct": round(100 * proba[k], 1),
                "summary": f"The model predicts {pred} with {proba[k]:.0%} confidence. An average row gets P({pred}) = "
                           f"{100 * base[k]:.1f}%; this row's features move it to {100 * proba[k]:.1f}%.",
                "contributions": sorted(rows, key=lambda r: -abs(r["effect_pts"])),
                "method": "exact Shapley values vs. a training background; effects add up from baseline to final probability"}

    def get_feature_context(feature: str = ""):
        out = {}
        for f in [by_name[feature.casefold()]] if feature and feature.casefold() in by_name else feats:
            s, v = stats[f], clean[f]
            if s["type"] == "numeric":
                if pd.isna(v):
                    out[f] = {"value": "missing", "note": f"imputed with the training median {s['median']:.4g}",
                              "summary": f"{f} is missing in this row; the model used the training median {_fmt(s['median'])}. Caveat: missing value."}
                    continue
                side = "above" if v > s["median"] else "below" if v < s["median"] else "equal to"
                q = pct(f, v)
                level = ("unusually low" if q < 10 else "low" if q < 30 else "typical" if q <= 70 else "high" if q <= 90 else "unusually high")
                by_cls = "; ".join(f"class {c} median {_fmt(m)} ({'above' if v > m else 'below' if v < m else 'equal'})"
                                   for c, m in s["median_by_class"].items())
                outside = bool(v < s["min"] or v > s["max"])
                out[f] = {"value": round(float(v), 4), "percentile_in_training": round(pct(f, v)),
                          "training_median": round(float(s["median"]), 4), "median_by_class": s["median_by_class"],
                          "outside_training_range": outside,
                          "summary": f"{f} = {_fmt(v)} is {level} ({_nth(q)} percentile of training data), {side} the overall "
                                     f"median {_fmt(s['median'])}. Compared with each class: {by_cls}. "
                                     + ("Caveat: OUTSIDE the training range." if outside else "Within the training range.")}
            else:
                key = "missing" if pd.isna(v) else v
                mix = s["class_mix"].get(key)
                out[f] = {"value": key, "share_of_training_rows": s["share"].get(key, 0.0),
                          "class_mix_for_this_value": mix if mix is not None else "unseen in training",
                          "overall_class_mix": {str(c): round(p, 3) for c, p in bundle["class_mix"].items()},
                          "summary": (f"{f} = {key}: never seen in training. Caveat: unseen category." if mix is None else
                                      f"{f} = {key}: {s['share'].get(key, 0):.0%} of training rows; {mix.get(pred, 0):.0%} of them are "
                                      f"class {pred} (vs {bundle['class_mix'].get(pred, 0):.0%} overall).")}
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
                          "summary": f"Changing {f} from {_fmt(clean[f])} to {_fmt(tries[i][1])} would flip the prediction to "
                                     f"{classes[P[i].argmax()]} ({P[i].max():.0%}).",
                          "size_of_change": f"{abs(pct(f, tries[i][1]) - cur):.0f} percentile points" if stats[f]["type"] == "numeric" else "category swap"})
        if flips:
            return {"flips": flips[:4]}
        i = int(P[:, k].argmin())
        return {"flips": [], "note": "no single-feature change flips the prediction",
                "summary": f"No single-feature change flips the prediction from {pred}; the prediction is robust.",
                "largest_drop": {"feature": tries[i][0], "to": _fmt(tries[i][1]), f"P({pred})": round(float(P[i, k]), 3)}}

    def what_if(changes: dict):
        known = {by_name[str(f).casefold()]: v for f, v in (changes or {}).items() if str(f).casefold() in by_name}
        p2 = score([{**raw, **known}])[0]
        new = classes[int(p2.argmax())]
        return {"applied_changes": known, "ignored": [f for f in (changes or {}) if str(f).casefold() not in by_name],
                "new_prediction": new, "new_probabilities": {c: round(float(q), 3) for c, q in zip(classes, p2)},
                "summary": f"With {', '.join(f'{f} = {v}' for f, v in known.items()) or 'no valid changes'}, the model predicts "
                           f"{new} ({p2.max():.0%}); as uploaded it predicted {pred} ({proba[k]:.0%})."}

    def get_model_card():
        lb = bundle["leaderboard"].set_index("Model")
        return {"model": bundle["model_name"], "classes": classes,
                "cv_accuracy": round(float(lb.loc[bundle["model_name"], "CV accuracy"]), 4),
                "test_accuracy": round(float(bundle["test"]["accuracy"]), 4),
                "top_features_globally": bundle["importance"]["Feature"].head(5).tolist(),
                "summary": f"The model is {bundle['model_name']}; it is {bundle['test']['accuracy']:.1%} accurate on the held-out "
                           f"test set. Most important features overall: {', '.join(bundle['importance']['Feature'].head(3))}."}

    tools = {f.__name__: f for f in (get_feature_contributions, get_feature_context, find_counterfactual, what_if, get_model_card)}
    facts = {"row": {f: _fmt(clean[f]) for f in feats}, "predicted_class": pred, "confidence": round(float(proba[k]), 3),
             "probabilities": {c: round(float(q), 3) for c, q in zip(classes, proba)}}
    return tools, facts


def _summaries(o) -> list:
    """All 'summary' sentences in a tool result, in order."""
    if isinstance(o, dict):
        return ([o["summary"]] if "summary" in o else []) + [t for k, v in o.items() if k != "summary" for t in _summaries(v)]
    return [t for v in o for t in _summaries(v)] if isinstance(o, list) else []


def _dumps(o):
    return json.dumps(o, default=lambda v: v.item() if hasattr(v, "item") else str(v), separators=(",", ":"))


def _ollama(path, payload=None, timeout=180):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    host = host if "://" in host else "http://" + host  # OLLAMA_HOST is often written as "127.0.0.1:11434"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["OLLAMA_API_KEY"]
    req = urllib.request.Request(host + path, json.dumps(payload).encode() if payload is not None else None, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_status():
    """(model, note): the model the agent will use, or None and why not."""
    try:
        names = [m["name"] for m in _ollama("/api/tags", timeout=3).get("models", []) if "embed" not in m["name"]]
    except Exception:
        return None, "Ollama is not running"
    if os.environ.get("OLLAMA_MODEL"):
        return os.environ["OLLAMA_MODEL"], ""
    pick = next((n for pref in (DEFAULT_MODEL, *MODEL_PREFS) for n in names if n.startswith(pref)), names[0] if names else None)
    return (pick, "") if pick else (None, f"Ollama has no model yet - run `ollama pull {DEFAULT_MODEL}`")


def ollama_chat(model, messages):
    """One /api/chat round with the agent's tools available; returns the assistant message."""
    msg = _ollama("/api/chat", {"model": model, "messages": messages, "tools": TOOL_SPECS, "stream": False,
                                "options": {"temperature": 0.2, "num_ctx": 8192}})["message"]
    msg["content"] = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
    return msg


def run_agent(tools, messages, chat, max_steps=5):
    """Tool-calling loop. `chat(messages)` returns an assistant message in Ollama's /api/chat shape."""
    trace = []
    for step in range(max_steps):
        if step == max_steps - 1:
            messages.append({"role": "user", "content": "Write the final answer now, without further tool calls."})
        msg = chat(messages)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", ""), trace
        if step == max_steps - 1:
            break
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            try:
                args = json.loads(args) if isinstance(args, str) else args
                result = tools[name](**args)
            except Exception as e:  # bad tool name/arguments go back to the model, not to the user
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": "\n".join(_summaries(result)) or _dumps(result)})
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


CLAIM = re.compile(r"`?(\w+)\s*=\s*([^\s`]+)`?\s+(raises|lowers|increases|decreases|reduces)\s+P\((\w+)\)\s+by\s+"
                   r"([\d.]+)\s*(?:probability\s+)?points?", re.I)


def fix_claims(text: str, contrib: dict) -> str:
    """Guardrail: an 'X = v raises/lowers P(c) by n points' claim that contradicts the Shapley values is rewritten."""
    truth = {r["feature"].casefold(): r for r in contrib["contributions"]}

    def check(m):
        r = truth.get(m.group(1).casefold())
        if r is None or m.group(4) != contrib["explaining_class"]:
            return m.group(0)
        same_way = (m.group(3).lower() in ("raises", "increases")) == (r["effect_pts"] > 0)
        return m.group(0) if same_way and abs(float(m.group(5)) - abs(r["effect_pts"])) < 0.15 else r["summary"]
    return CLAIM.sub(check, text)


def _evidence(tools):
    """The agent's fixed first step for any row: drivers, training-data context, what would flip it."""
    return [(n, tools[n]()) for n in ("get_feature_contributions", "get_feature_context", "find_counterfactual")]


def _briefing(facts, evidence) -> str:
    return (f"Prediction: {facts['predicted_class']} with {facts['confidence']:.0%} confidence.\nVerified evidence:\n"
            + "\n".join(f"- {t}" for _, r in evidence for t in _summaries(r)))


def explain(bundle: dict, x: pd.DataFrame, model: str | None = None, chat=None) -> dict:
    tools, facts = make_tools(bundle, x)
    evidence = _evidence(tools)
    (_, contrib), (_, ctx), (_, cf) = evidence
    trace = [{"tool": n, "args": {}, "result": r} for n, r in evidence]
    text, engine = "", "Offline explainer (Ollama not available) - same tools, templated wording"
    if model:
        try:
            text, more = run_agent(tools, [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": _briefing(facts, evidence) + "\n\nWrite the explanation."}],
                                   chat or (lambda m: ollama_chat(model, m)))
            trace += more
            text = fix_claims(text, contrib)
            # caveats are facts, so they come from the data, never from the model
            text = re.sub(r"\n*\**Caveats?\**.*\Z", "", text, flags=re.S | re.I).rstrip()
            caveats = [f"`{f}`: {c['summary'].split('Caveat: ')[1].rstrip('.')}" for f, c in ctx.items() if "Caveat: " in c["summary"]]
            caveats += [f"low confidence ({facts['confidence']:.1%}) - close to the decision boundary"] if facts["confidence"] < 0.6 else []
            text += ("\n\n**Caveats** - " + "; ".join(caveats) + ".") if caveats and text else ""
            engine = f"Ollama · {model} (agent)" if text else "Offline explainer - the model gave no final answer"
        except Exception as e:
            engine = f"Offline explainer - Ollama call failed ({type(e).__name__}: {str(e)[:160]})"
    if not text:
        text = offline_text(facts, contrib, ctx, cf)
    return {"text": text, "engine": engine, "trace": trace, "contrib": contrib, "facts": facts}


def ask(bundle: dict, x: pd.DataFrame, explanation: str, history: list, question: str, model: str | None = None, chat=None) -> dict:
    """Answer a follow-up question about one row. `history` = earlier turns as [{"role", "content"}]."""
    tools, facts = make_tools(bundle, x)
    note = ""
    asked = asked_changes(question, facts["row"])
    if asked:  # guardrail: a what-if re-scores exactly the values the user named, whatever the model passes
        what_if = tools["what_if"]
        tools = {**tools, "what_if": lambda changes=None: what_if(asked)}
    if model:
        msgs = [{"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": _briefing(facts, _evidence(tools))},
                {"role": "assistant", "content": explanation}, *history, {"role": "user", "content": question}]
        try:
            text, trace = run_agent(tools, msgs, chat or (lambda m: ollama_chat(model, m)))
            if text:
                return {"text": fix_claims(text, tools["get_feature_contributions"]()), "tools": [t["tool"] for t in trace]}
            note = "the model gave no answer"
        except Exception as e:
            note = f"Ollama call failed ({type(e).__name__})"
    return offline_answer(question, tools, facts) | ({"note": note} if note else {})


def asked_changes(question: str, features) -> dict:
    """Explicit feature values in a question: 'what if A were 40 and C = north' -> {'A': '40', 'C': 'north'}."""
    feats = {f.casefold(): f for f in features}
    pairs = re.findall(r"\b(\w+)\s*(?:==|=|:|\bto\b|\bwere\b|\bwas\b|\bis\b|\bat\b)\s*(\"[^\"]*\"|'[^']*'|[-\w.,]+)", question)
    return {feats[f.casefold()]: v.strip("\"'").rstrip(".,") for f, v in pairs if f.casefold() in feats}


def offline_answer(question: str, tools: dict, facts: dict) -> dict:
    """No LLM: answer what-if and what-would-flip-it questions straight from the tools."""
    changes = asked_changes(question, facts["row"])
    if changes:
        r = tools["what_if"](changes)
        probs = ", ".join(f"{c} {p:.0%}" for c, p in r["new_probabilities"].items())
        return {"text": f"With {', '.join(f'`{k} = {v}`' for k, v in r['applied_changes'].items())} the model predicts "
                        f"**{r['new_prediction']}** ({probs}); as uploaded it was **{facts['predicted_class']}** "
                        f"({facts['confidence']:.0%}).", "tools": ["what_if"]}
    if re.search(r"flip|change|different|other class|would it take", question, re.I):
        cf = tools["find_counterfactual"]()
        lines = [f"- `{f['feature']}` {f['from']} -> {f['to']} flips it to **{f['new_prediction']}** ({f['new_confidence']:.0%})"
                 for f in cf["flips"]]
        return {"text": "Single changes that would flip the prediction:\n" + "\n".join(lines) if lines
                else "No single-feature change flips this prediction - it is robust.", "tools": ["find_counterfactual"]}
    return {"text": "The local model isn't running, so right now I can only answer what-if questions "
                    "(for example `what if A = 40?`) and *what would flip it?*.", "tools": []}
