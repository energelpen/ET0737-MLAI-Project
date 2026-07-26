# ET0737 Machine Learning & AI — Mini Project
## Diabetes Risk Prediction from the CDC BRFSS 2015 Health Survey

**School of EEE, Singapore Polytechnic — AY26/27 S1**

Predict whether a person is **at risk of diabetes / pre-diabetes** from cheap,
self-reported health-survey answers (no blood test) — a low-cost pre-screen that
tells a clinic *who should be referred for a confirmatory glucose test*.

- **Task type:** binary classification (imbalanced, ~16 % positive)
- **Dataset:** CDC *Behavioral Risk Factor Surveillance System (BRFSS) 2015* — 441,456 US adults
  - Raw source: https://www.cdc.gov/brfss/annual_data/annual_2015.html
  - Kaggle mirror: https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system

---

## Repository layout

```
ET0737-MLAI-Project/
├── README.md
├── requirements.txt
├── ET0737_Diabetes_Risk_Prediction_Slides.pptx   # presentation deck
├── data/
│   ├── raw/
│   │   ├── 2015.csv.zip                    # raw BRFSS source archive (79 MB)
│   │   └── 2015.csv                        # extracted raw survey, 441k x 330 (517 MB, git-ignored)
│   └── processed/
│       └── diabetes_2015_clean.csv         # tidy 240,184 x 34 modelling table
├── notebooks/
│   ├── 01_Diabetes_2015_Cleaning.ipynb     # raw survey  ->  clean modelling table + EDA
│   └── 02_Diabetes_2015_Modelling.ipynb    # full ML workflow (EDA -> train -> evaluate -> export)
├── models/                                 # deployment artifacts written by notebook 02 §12
│   ├── diabetes_stack_recall_first.joblib      # the whole fitted pipeline
│   └── diabetes_stack_recall_first.meta.json   # threshold, feature order, library versions
├── clinic_app/                             # companion Flask app that serves the exported model
└── run_clinic.py                           # entry point for the clinic app
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order (each is self-contained and uses relative paths):

1. **`01_Diabetes_2015_Cleaning.ipynb`** — reads `data/raw/2015.csv`, recodes ~26 raw
   BRFSS columns into 33 clean, medically-motivated risk factors + a 3-stage diabetes
   target, drops *Don't-know / Refused* codes, and writes `data/processed/diabetes_2015_clean.csv`.
   *(The clean CSV is already committed, so you can skip straight to notebook 02.)*

2. **`02_Diabetes_2015_Modelling.ipynb`** — the full machine-learning workflow:
   - EDA & feature justification (Matplotlib / Seaborn)
   - leakage-safe preprocessing: `StandardScaler` + `OneHotEncoder` inside a `Pipeline`,
     with a **60 / 20 / 20 stratified train / validation / test** split
   - **six models spanning five model families**, each hyperparameter-tuned with Grid /
     Randomized Search CV: **Logistic Regression** (linear), **Random Forest** (bagged trees),
     **SVM (RBF)** (kernel), a **Neural Network / MLP** (`§5.4`, with its training curve) and
     **XGBoost** (boosted trees), plus a **stacking ensemble** that combines the two strongest
     base learners (**SVM + XGBoost**) under a **Logistic Regression meta-learner** (`§5.6`)
   - evaluation on Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrices, a 5-fold
     cross-validated overfitting (train-vs-validation) check, and a one-touch held-out **test** evaluation
   - **§6.5 — operating-threshold tuning** that raises accuracy *and* recall together by
     choosing the decision threshold on validation (max-F1 / max balanced-accuracy / high-recall)
   - **§6.6 — the deployed model:** the stacking ensemble at a **recall-first** operating point
     (recall in the 0.80–0.85 safety band), so at-risk patients are rarely missed
   - new-patient prediction + an interactive `ipywidgets` risk calculator
   - business insights, a Responsible-GenAI reflection, and a **deployment export** (`§12`):
     the whole fitted pipeline saved as one `joblib` artifact + a `.meta.json` sidecar, loaded
     directly by the companion `clinic_app/` web app

   > **Honest performance.** ROC-AUC on these cheap survey features lands around **0.82** — the
   > well-documented BRFSS ceiling. Some published studies report ~98 % by applying SMOTE to the
   > *whole* dataset *before* the train/test split, which leaks synthetic test rows into training;
   > we avoid that and report the honest, deployable figure.

## Notes on reproducibility & resource use

- All randomness is seeded with `RANDOM_STATE = 42`.
- The modelling notebook caps parallelism (`N_JOBS = 4`) and keeps each estimator
  single-threaded so cross-validation never spawns nested worker pools that would copy the
  144k-row training set and exhaust memory. Full run is ~10–15 min on an 8-core / 16 GB machine.
- The RBF-SVM is tuned on a stratified 15k subsample (a full RBF-SVM is O(n²) and
  impractical on 144k rows) — documented as an explicit engineering trade-off.
