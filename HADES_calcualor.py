"""
HADES Score Predictor
=====================
Predicts HADES scores for molecules using an ensemble of classifiers.
Models are loaded from Figshare (article 32084259) and applied to
molecular descriptors computed from SMILES strings.

Usage:
    1. Set your SMILES list (or point to a CSV) in the CONFIG section below.
    2. Run:  python hades_score.py
    3. Results are printed to the console and saved to hades_results.csv.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import json
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from HADES.featurize import Featurizer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit this section only
# ══════════════════════════════════════════════════════════════════════════════

# Option A: provide SMILES directly
SMILES_LIST = [
    "NC(=O)N1CCN(C(=O)Cc2ccc(CN3CCc4ccccc43)cc2)CC1",
    "Cc1cccnc1NC(=O)c1cccc(C(=O)N2CCN(C)CC2)c1",
]

# Option B: load from a CSV (set CSV_PATH to your file; leave None to use Option A)
#   The column containing SMILES must be named "SMILES".
CSV_PATH = None  # e.g. "my_molecules.csv"

# Paths to local support files
HYPERPARAMS_JSON = "./HADES_models_hyperparameters.json"
X_TRAIN_PARQUET  = "./train_test_saved/X_train.parquet"

# Figshare article that hosts the pre-trained model .pkl files
FIGSHARE_ARTICLE_ID = 32084259

# Where to write results
OUTPUT_CSV = "hades_results.csv"

# ══════════════════════════════════════════════════════════════════════════════


def load_smiles() -> list[str]:
    """Return the SMILES list from a CSV or from SMILES_LIST."""
    if CSV_PATH:
        print(f"📂  Loading SMILES from '{CSV_PATH}' …")
        df = pd.read_csv(CSV_PATH)
        if "SMILES" not in df.columns:
            sys.exit("❌  CSV must contain a column named 'SMILES'.")
        smiles = df["SMILES"].dropna().tolist()
        print(f"    {len(smiles)} molecules found.")
        return smiles
    print(f"🔬  Using {len(SMILES_LIST)} hard-coded SMILES.")
    return SMILES_LIST


def build_model_shells(hyperparams_path: str) -> dict:
    """Instantiate classifier shells with tuned hyper-parameters (no weights yet)."""
    print(f"\n⚙️   Loading hyper-parameters from '{hyperparams_path}' …")
    with open(hyperparams_path) as fh:
        results = json.load(fh)

    best = {name: vals["best_params"] for name, vals in results.items()}

    return {
        "XGBClassifier": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **best["XGBClassifier"]
        ),
        "CatBoostClassifier": CatBoostClassifier(
            verbose=0, **best["CatBoostClassifier"]
        ),
        "LGBMClassifier": LGBMClassifier(**best["LGBMClassifier"]),
        "RandomForestClassifier": RandomForestClassifier(
            **best["RandomForestClassifier"]
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            **best["HistGradientBoostingClassifier"]
        ),
    }


def load_models_from_figshare(model_shells: dict, article_id: int) -> dict:
    """Download fitted .pkl models from Figshare and return them in a dict."""
    print(f"\n🌐  Fetching model list from Figshare article {article_id} …")
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    response = requests.get(api_url)
    response.raise_for_status()

    url_map = {
        f["name"].replace(".pkl", ""): f["download_url"]
        for f in response.json()
        if f["name"].endswith(".pkl")
    }

    loaded = {}
    print("⬇️   Downloading models:")
    for name in model_shells:
        if name not in url_map:
            print(f"  ⚠️  '{name}' not found in Figshare — skipping.")
            continue
        print(f"  • {name} …", end=" ", flush=True)
        r = requests.get(url_map[name])
        r.raise_for_status()
        loaded[name] = joblib.load(io.BytesIO(r.content))
        print("✓")

    if not loaded:
        sys.exit("❌  No models could be loaded. Aborting.")

    return loaded


def featurize(smiles: list[str], descriptor_cols: list[str]) -> pd.DataFrame:
    """Convert SMILES to numerical descriptors used during training."""
    print("\n🧪  Featurizing molecules …")
    featurizer = Featurizer()
    df = featurizer.featurize_many_smiles(smiles)
    df.columns = df.columns.astype(str)
    df[descriptor_cols] = df[descriptor_cols].apply(pd.to_numeric, errors="coerce")
    print(f"    Feature matrix: {df.shape[0]} molecules × {len(descriptor_cols)} descriptors.")
    return df


def predict_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Average predicted probabilities across all ensemble members."""
    probs = [m.predict_proba(X)[:, 1] for m in models.values()]
    return np.mean(probs, axis=0)


def display_results(smiles: list[str], scores: np.ndarray) -> pd.DataFrame:
    """Pretty-print results and return a tidy DataFrame."""
    results_df = pd.DataFrame({"SMILES": smiles, "HADES_Score": scores})

    print("\n" + "═" * 60)
    print("  HADES Score Results")
    print("═" * 60)
    col_w = min(50, max(len(s) for s in smiles) + 2)
    print(f"  {'SMILES':<{col_w}}  HADES Score")
    print("  " + "-" * (col_w + 14))
    for _, row in results_df.iterrows():
        smi = row["SMILES"]
        if len(smi) > col_w - 1:
            smi = smi[: col_w - 4] + "…"
        print(f"  {smi:<{col_w}}  {row['HADES_Score']:.4f}")
    print("═" * 60)

    return results_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load SMILES
    smiles = load_smiles()

    # 2. Load descriptor column names from training set
    print(f"\n📊  Reading training descriptor columns from '{X_TRAIN_PARQUET}' …")
    X_train = pd.read_parquet(X_TRAIN_PARQUET)
    descriptor_cols = list(X_train.columns)[:-1]   # last column is the label

    # 3. Build model shells with tuned hyper-parameters
    model_shells = build_model_shells(HYPERPARAMS_JSON)

    # 4. Download fitted models from Figshare
    models = load_models_from_figshare(model_shells, FIGSHARE_ARTICLE_ID)

    # 5. Featurize input molecules
    features = featurize(smiles, descriptor_cols)

    # 6. Predict
    print(f"\n🤖  Running ensemble prediction ({len(models)} models) …")
    scores = predict_ensemble(models, features[descriptor_cols])

    # 7. Display & save
    results_df = display_results(smiles, scores)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾  Results saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()