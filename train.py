"""Train from the command line:  python train.py data/<client_file>.csv|.xlsx

Writes artifacts/model.joblib (pipeline + report) and artifacts/holdout_test*.csv (the 10% test set).
"""
import sys
from pathlib import Path

import pipeline

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_demo.csv")
b = pipeline.train(pipeline.read_table(path.read_bytes(), path.name), source=path.name,
                   progress=lambda frac, msg: print(f"[{frac:4.0%}] {msg}", flush=True))
print(b["log"].to_string(index=False), "\n")
print(b["leaderboard"].drop(columns="Best params").round(4).to_string(index=False))
t = b["test"]
print(f"\nSelected {b['model_name']}: test accuracy {t['accuracy']:.4f} (baseline {t['baseline_accuracy']:.4f}), "
      f"macro-F1 {t['f1_macro']:.4f}, ROC-AUC {t['roc_auc']:.4f}")
