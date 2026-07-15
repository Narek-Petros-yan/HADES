# HADES — Holistic AI-based Drug-likeness Estimation Score

HADES is a machine learning–based soft voting ensemble for predicting approved oral drug-likeness. It is trained to distinguish approved oral drugs from a curated non-drug set, and its predictions reflect the likelihood that a compound resembles an approved oral drug — not a general measure of drug-likeness.
---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Overview

HADES is a machine learning–based soft voting ensemble for predicting approved oral drug-likeness. It is trained to distinguish approved oral drugs from a curated non-drug set, and its predictions reflect the likelihood that a compound resembles an approved oral drug — not a general measure of drug-likeness.
The ensemble integrates five complementary gradient boosting and tree-based algorithms — Random Forest, HistGradient Boosting, XGBoost, CatBoost, and LightGBM — by averaging their predicted probabilities. Molecular features are drawn from two sources:

Physicochemical descriptors computed with Mordred
ADMET properties predicted by ADMET-AI

Feature importance analysis identified BCUTd-1l (the first lowest eigenvalue of the Burden matrix weighted by sigma electrons) as the most influential physicochemical feature, and CYP2C19_Veith as the most important ADMET feature in the final ensemble.
A scaffold-split variant, HADES_scs, is also included to provide additional insight into model behaviour under more stringent data partitioning. Its results across stress tests were broadly consistent with the main model, supporting the robustness of the ensemble architecture.
Key capabilities

Clinical stage discrimination — compounds progressing to later clinical stages receive higher HADES scores, with ROC-AUC superior to QED, MolSkill, and DBPP-SCORE
Temporal robustness — correctly assigns high scores to drugs approved after 2024 (time-split validation)
Chemical specificity — assigns lower scores to small-ring-containing, orally toxic, and unusual-valency compounds
Hit/lead optimization — in curated medicinal chemistry cases, optimized molecules consistently received higher scores than their starting points
Virtual screening — effectively prioritizes drug-like candidates within targeted screening libraries


Note: HADES is designed to complement, not replace, other drug-likeness metrics. Users are encouraged to apply multiple scoring functions in conjunction and perform context-specific evaluation before relying on any single score.


---

## Installation

conda env create -f HADES.yml
conda activate hades

The HADES.yml file contains all required libraries with strict version specifications.
During installation, you may need to manually install certain packages with the exact versions specified.
---

## Project Structure

```
.
├── catboost_info/                        # CatBoost training logs and metadata
├── compounds dataset/                    # Raw compound datasets
├── HADES/                                # Core HADES module / source code
├── scaffold_split/                       # Scaffold-based train/test splits
├── screening libraries all scores/       # Scoring results for screening libraries
├── seed_dependency/                      # Seed dependency analysis outputs
├── seed_depenndency_balanced/            # Balanced seed dependency outputs
├── train_test_saved/                     # Saved train/test datasets for random split
│
├── HADES_calcualor.py                    # Main HADES score calculator
├── HADES_scs_calculator.py               # HADES_scs (scaffold splitted) calculator
├── HADES_models_hyperparameters.json     # Hyperparameters for all models for HADES
├── HADES.yml                             # Conda environment file
│
├── initial_data_for_HADES_pipeline.csv   # Input data for the full pipeline
├── clinical_candidates_test_scores.csv   # HADES scores for clinical candidates
├── Hit_lead_optimization_data.csv        # Hit-to-lead optimization dataset
│
├── HADES pipeline.ipynb                  # End-to-end pipeline notebook
├── scaffold split for models.ipynb       # Scaffold splitting methodology
└── SEED dependency on feature selection.ipynb  # SEED feature selection analysis
```

---

## Usage

### Running the HADES Calculator

```bash
python HADES_calculator.py --input <input_file.csv> --output <output_file.csv>
```

### Running the SCS Calculator

```bash
python HADES_scs_calculator.py --input <input_file.csv>
```

### Full Pipeline

Open and run the notebook:

```bash
jupyter notebook "HADES pipeline.ipynb"
```

---

## License

[MIT]
