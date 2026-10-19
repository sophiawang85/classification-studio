"""End-to-end smoke test:  python test_smoke.py   (~15 s, trains a tiny model in a temp dir)"""
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import agent
import pipeline as pl

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({"A": rng.normal(size=n), "B": rng.choice(["x", "y", "z"], n), "C": rng.normal(size=n),
                   "D": rng.integers(0, 5, n), "E": rng.normal(size=n), "F": rng.choice(["u", "v"], n), "G": rng.normal(size=n)})
df["H"] = np.where(df.A + (df.B == "x") + rng.normal(scale=0.5, size=n) > 0.5, "yes", "no")
df = df.astype({"C": object})
df.loc[:20, "C"] = "?"

with tempfile.TemporaryDirectory() as tmp:
    b = pl.train(df, out_dir=Path(tmp), fast=True)
    assert b["test"]["accuracy"] > 0.7, b["test"]["accuracy"]
    assert (Path(tmp) / "holdout_test.csv").exists() and len(pd.read_csv(Path(tmp) / "holdout_test.csv")) == 40

    # messy upload: lower-case + reordered headers, missing G, extra column, unseen category, text in a number column
    up = df.drop(columns=["H", "G"]).head(30).rename(columns=str.lower).assign(extra=1)[["f", "e", "d", "c", "b", "a", "extra"]]
    up.loc[0, "b"] = "brand-new"
    up = up.astype({"a": object})
    up.loc[1, "a"] = "not a number"
    X, y, issues = pl.align(up, b)
    out, proba = pl.predict(b, X)
    text = " ".join(m for _, m in issues)
    assert len(out) == 30 and y is None and np.allclose(proba.sum(1), 1)
    assert "Missing column(s) G" in text and "brand-new" in text and "not a number" in text and "extra" in text, text

    # file-level edge cases through the real reader
    csv = df.drop(columns="H").head(10).to_csv(index=False)
    for name, data in {"semicolon.csv": csv.replace(",", ";").encode(),
                       "noheader.csv": csv.split("\n", 1)[1].encode(),
                       "latin1.csv": csv.replace("x", "é").encode("latin-1")}.items():
        X, _, issues = pl.align(pl.read_table(data, name), b)
        assert len(pl.predict(b, X)[0]) == 10, name
    xl = io.BytesIO()
    df.head(10).to_excel(xl, index=False)
    X, y, _ = pl.align(pl.read_table(xl.getvalue(), "f.xlsx"), b)
    assert y is not None and len(pl.predict(b, X)[0]) == 10
    for name, data in {"empty.csv": b"", "header_only.csv": b"A,B,C\n", "junk.csv": b"x,y\n1,2\n"}.items():
        try:
            pl.align(pl.read_table(data, name), b)
            raise AssertionError(f"{name} should have been rejected")
        except ValueError:
            pass

    # Shapley values add up exactly from the baseline to the prediction
    base, phi, fx = agent.shapley(b["pipeline"], X.iloc[[2]], b["background"], b["features"])
    assert np.allclose(base + phi.sum(0), fx) and np.allclose(fx, b["pipeline"].predict_proba(X.iloc[[2]])[0])

    # offline explainer always answers
    e = agent.explain(b, X.iloc[[2]])
    assert "**Verdict**" in e["text"] and e["trace"][0]["tool"] == "get_feature_contributions"

    # agent loop with a scripted LLM (Ollama /api/chat message shape): one tool call, then a final answer
    replies = iter([{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_feature_contributions", "arguments": {}}}]},
                    {"role": "assistant", "content": "**Verdict** - because of A."}])
    e2 = agent.explain(b, X.iloc[[2]], model="fake", chat=lambda msgs: next(replies))
    assert e2["text"].startswith("**Verdict**") and "fake" in e2["engine"] and e2["trace"][0]["tool"] == "get_feature_contributions"

    # follow-up chat: scripted LLM answer, then the offline fallback for what-if / what-would-flip-it questions
    a = agent.ask(b, X.iloc[[2]], e["text"], [], "why?", model="fake", chat=lambda msgs: {"role": "assistant", "content": "Because of A."})
    assert a["text"] == "Because of A." and a["tools"] == []
    assert "predicts" in agent.ask(b, X.iloc[[2]], e["text"], [], "what if A were 3?")["text"]
    assert "flip" in agent.ask(b, X.iloc[[2]], e["text"], [], "what would flip it?")["text"]

    # guardrail: a contribution claim with the wrong direction is rewritten from the Shapley values
    r0 = e["contrib"]["contributions"][0]
    wrong = f"{r0['feature']} = {r0['value']} {'lowers' if r0['effect_pts'] > 0 else 'raises'} P({e['contrib']['explaining_class']}) by {abs(r0['effect_pts'])} points"
    assert agent.fix_claims(wrong, e["contrib"]) == r0["summary"] and agent.fix_claims(r0["summary"], e["contrib"]) == r0["summary"]

print("ok - all smoke checks passed")
