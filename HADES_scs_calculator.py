"""
HADES_scs Score Calculator (Scaffold Split)
========================================
Trains an ensemble on scaffold-split data, then computes HADES scores
for your molecules.

Usage:
    1. Set your SMILES in the CONFIG section below (or point to a CSV).
    2. Run
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import sys
from collections import Counter
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from HADES.featurize import Featurizer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this section only
# ══════════════════════════════════════════════════════════════════════════════

# ── Input molecules ───────────────────────────────────────────────────────────
# Option A: provide SMILES directly
SMILES_LIST = [
    "NC(=O)N1CCN(C(=O)Cc2ccc(CN3CCc4ccccc43)cc2)CC1",
    "Cc1cccnc1NC(=O)c1cccc(C(=O)N2CCN(C)CC2)c1",
]

# Option B: load from CSV (set path; column must be named "SMILES")
CSV_PATH = None  # e.g. "my_molecules.csv"

# ── Training data & hyper-parameters ─────────────────────────────────────────
SCAFFOLD_DIR     = Path("./scaffold_split")
X_TRAIN_PATH     = SCAFFOLD_DIR / "x_train_scaffold_split"
Y_TRAIN_PATH     = SCAFFOLD_DIR / "y_train_scaffold_split.csv"
HYPERPARAMS_JSON = SCAFFOLD_DIR / "optuna_results_final_scaffold_split.json"

PHASE_COLUMN   = "phase"
PHASE_DIVISOR  = 4        # int(phase / 4) → binary label
CLASS_WEIGHT_SCHEME = "balanced"
RANDOM_SEED    = 42
N_TRAILING_NON_FEATURE_COLS = 2

# ── Optional: save trained models ────────────────────────────────────────────
SAVE_MODELS     = False           # set True to persist
SAVE_MODELS_PATH = "trained_models.pkl"

# ══════════════════════════════════════════════════════════════════════════════


def load_smiles() -> list[str]:
    if CSV_PATH:
        print(f"📂  Loading SMILES from '{CSV_PATH}' …")
        df = pd.read_csv(CSV_PATH)
        if "SMILES" not in df.columns:
            sys.exit("❌  CSV must contain a column named 'SMILES'.")
        return df["SMILES"].dropna().tolist()
    return SMILES_LIST


def load_training_data():
    print("📂  Loading training data …")
    X_train = pd.read_parquet(X_TRAIN_PATH)
    y_raw   = pd.read_csv(Y_TRAIN_PATH)
    descriptor_cols = list(X_train.columns)[:-N_TRAILING_NON_FEATURE_COLS]
    y = [int(v / PHASE_DIVISOR) for v in y_raw[PHASE_COLUMN]]
    return X_train[descriptor_cols], y, descriptor_cols


def build_and_train(X_train, y_train) -> dict:
    print(f"\n⚙️   Loading hyper-parameters …")
    with open(HYPERPARAMS_JSON) as fh:
        results = json.load(fh)

    best = {}
    for name, values in results.items():
        params = dict(values["best_params"])
        params["random_seed" if name == "CatBoostClassifier" else "random_state"] = RANDOM_SEED
        best[name] = params

    counts = Counter(y_train)
    total  = sum(counts.values())
    cb_weights    = [total / counts[c] for c in sorted(counts)]
    sample_weights = compute_sample_weight(class_weight=CLASS_WEIGHT_SCHEME, y=y_train)

    models = {
        "XGBClassifier": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **best["XGBClassifier"]
        ),
        "CatBoostClassifier": CatBoostClassifier(
            verbose=0, class_weights=cb_weights, **best["CatBoostClassifier"]
        ),
        "LGBMClassifier": LGBMClassifier(**best["LGBMClassifier"]),
        "RandomForestClassifier": RandomForestClassifier(**best["RandomForestClassifier"]),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            **best["HistGradientBoostingClassifier"]
        ),
    }

    trained = {}
    print(f"🏋️   Training {len(models)} models …")
    for name, model in tqdm(models.items(), desc="  Progress", unit="model"):
        if name == "CatBoostClassifier":
            trained[name] = model.fit(X_train, y_train)
        else:
            trained[name] = model.fit(X_train, y_train, sample_weight=sample_weights)

    return trained


def featurize(smiles: list[str], descriptor_cols: list[str]) -> pd.DataFrame:
    print("\n🧪  Featurizing molecules …")
    df = Featurizer().featurize_many_smiles(smiles)
    df.columns = df.columns.astype(str)
    df[descriptor_cols] = df[descriptor_cols].apply(pd.to_numeric, errors="coerce")
    return df


def predict_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    probs = [m.predict_proba(X)[:, 1] for m in models.values()]
    return np.mean(probs, axis=0)


def display_results(smiles: list[str], scores: np.ndarray):
    print("\n" + "═" * 58)
    print("  HADES Score Results")
    print("═" * 58)
    col_w = min(48, max(len(s) for s in smiles) + 2)
    print(f"  {'SMILES':<{col_w}}  HADES Score")
    print("  " + "─" * (col_w + 13))
    for smi, score in zip(smiles, scores):
        label = smi if len(smi) <= col_w else smi[:col_w - 3] + "…"
        print(f"  {label:<{col_w}}  {score:.4f}")
    print("═" * 58)


def main():
    smiles = load_smiles()
    X_train, y_train, descriptor_cols = load_training_data()
    models = build_and_train(X_train, y_train)

    if SAVE_MODELS:
        joblib.dump(models, SAVE_MODELS_PATH)
        print(f"\n💾  Models saved to '{SAVE_MODELS_PATH}'.")

    features = featurize(smiles, descriptor_cols)

    print(f"\n🤖  Computing HADES scores ({len(models)} models) …")
    scores = predict_ensemble(models, features[descriptor_cols])

    display_results(smiles, scores)


if __name__ == "__main__":
    main()