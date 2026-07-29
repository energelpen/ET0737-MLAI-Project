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
│   ├── diabetes_risk_model.joblib          # the whole fitted, calibrated estimator (written by §12a)
│   └── diabetes_risk_model.meta.json       # model name, threshold, feature order, library versions
├── clinic_app/                             # companion Flask app that serves the exported model
├── run_clinic.py                           # LOCAL entry point (Flask dev server, debug=True)
├── wsgi.py                                 # PRODUCTION entry point (gunicorn imports this)
├── requirements-app.txt                    # runtime-only deps for the web app (no jupyter/matplotlib)
└── render.yaml                             # Render Blueprint: build, start command, secrets
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
   - **nine models spanning eight model families**, each hyperparameter-tuned with Grid /
     Randomized Search CV and each built by **one shared `train_model()` helper** (`§5.0`), so
     every model provably gets identical preprocessing, CV protocol and scoring — the comparison
     is like-for-like by construction rather than by proof-reading:

     | § | Model | Family | The question it settles |
     |---|-------|--------|-------------------------|
     | 5.1 | Logistic Regression | linear | the interpretable baseline everything must beat |
     | 5.2 | Decision Tree | single tree | the control that makes "bagging/boosting is better" testable |
     | 5.3 | Random Forest | bagged trees | does averaging decorrelated trees pay? |
     | 5.4 | k-Nearest Neighbours | instance-based | is diabetes risk *local* — do similar answers share an outcome? |
     | 5.5 | SVM (RBF) | kernel | do smooth curved boundaries help? |
     | 5.6 | Neural Network / MLP (with its training curve) | neural | can a net learn interactions we didn't hand-craft? |
     | 5.7 | XGBoost | boosted trees, level-wise | the usual top performer on tabular health data |
     | 5.8 | LightGBM | boosted trees, leaf-wise | does a different tree *shape* change the answer? |
     | 5.9 | Stacking ensemble (XGBoost + RF + SVM + LR → LR meta-learner) | ensemble of four families | is a learned combination worth its complexity? |

     Two *independent* gradient-boosting implementations are deliberate: they disagree about how a
     tree is grown, so where both land on the same score the ceiling belongs to the **data**, not
     to a library.

     **Four further models were built, measured and then cut** — Extra Trees, HistGradientBoosting,
     Gaussian Naive Bayes and an SVM-on-PCA arm. Each landed on the same ROC-AUC as a model already
     on the board, so they cost runtime without adding evidence. That deletion is reported in `§11`
     as a result rather than hidden as a tidy-up: knowing which models to *remove* is the harder
     half of "just try more models".
   - **§5.10 — PCA as a diagnostic, not a tenth model.** It was first added as preprocessing (an
     SVM on principal components) and cut, because a principal component is a blend of all 48
     columns — "BMI" and "HighBP" stop being nameable inputs, which the §7 importance plots, the
     §7.1 fairness audit and the clinic app's per-factor explanations all depend on. What it is
     kept for is the picture: it shows geometrically why every model plateaus near ROC-AUC 0.83
   - evaluation on Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrices, a 5-fold
     cross-validated overfitting (train-vs-validation) check, and a one-touch held-out **test** evaluation
   - **§6.4 — a matched-recall comparison**, the fair way to rank models on an imbalanced
     target: hold recall at 0.80 for every model and compare precision, rather than comparing
     at an arbitrary 0.50 cut-off that means something different for each model
   - **§6.5 — operating-threshold tuning** that raises accuracy *and* recall together by
     choosing the decision threshold on validation (max-F1 / max balanced-accuracy / high-recall)
   - **§6.4b — the deployment decision, by a rule fixed in advance** rather than by picking a
     favourite: only models fitted on the *full* training split are eligible (the two SVM arms are
     tuned on a 15k subsample), the winner is the highest precision at matched recall, and an
     ensemble must clear a small **parsimony margin** to displace a single model. A paired
     bootstrap (`§7.2`) then tests the deployed model against its closest rival
   - **§6.6 — the deployed model** at a **recall-first** operating point (validation recall held
     in the 0.80–0.85 safety band), so at-risk patients are rarely missed
   - **§6.7 — probability calibration**, so the risk percentage shown to a patient is true and
     not merely well-ranked (the raw model was over-confident by ~2.5×)
   - **§7.1 — a fairness audit**: recall by race, sex, income and age band at the deployed
     threshold, reported as an equal-opportunity gap
   - **§7.2 — bootstrap confidence intervals** on the headline test metrics
   - new-patient prediction + an interactive `ipywidgets` risk calculator
   - business insights, a Responsible-GenAI reflection, and a **deployment export** (`§12`):
     the whole fitted pipeline saved as one `joblib` artifact + a `.meta.json` sidecar, loaded
     directly by the companion `clinic_app/` web app

   > **Running notebook 02 end-to-end is the whole build.** `§12c` is a *gate*, not a summary: it
   > re-reads both artifacts from disk, checks the sidecar against the notebook's own threshold and
   > feature order, verifies the reloaded model reproduces the deployed decisions, confirms the file
   > is inside GitHub's 100 MB limit, and loads it through `clinic_app`'s real code path. It
   > **raises** if any hard check fails, so an incomplete export cannot pass for a finished one.
   > When it prints `ALL HARD CHECKS PASSED`, `python run_clinic.py` serves exactly the model
   > evaluated above — nothing else to do by hand.
   >
   > **On artifact size.** `§6.4b`'s rule 3 keeps the deployed model small enough to commit, so a
   > fresh clone is runnable without re-running the notebook. That was a lesson, not a given: an
   > earlier run selected a Random Forest whose artifact was **~340 MB** — past the 100 MB limit,
   > bought for **+0.0004 precision** that `§7.2` cannot distinguish from zero. `§10` records that
   > episode, and that rule 3 was added *after* it. `§12a` prints the measured size and a
   > version-control verdict on every run.

   > **Honest performance.** ROC-AUC on these cheap survey features lands around **0.82** — the
   > well-documented BRFSS ceiling. Some published studies report ~98 % by applying SMOTE to the
   > *whole* dataset *before* the train/test split, which leaks synthetic test rows into training;
   > we avoid that and report the honest, deployable figure.

## Notes on reproducibility & resource use

- All randomness is seeded with `RANDOM_STATE = 42`.
- The modelling notebook caps parallelism (`N_JOBS = 4`) and keeps each estimator
  single-threaded so cross-validation never spawns nested worker pools that would copy the
  144k-row training set and exhaust memory. A full end-to-end run of notebook 02 takes roughly
  **25–40 min** on an 8-core / 16 GB machine (nine tuned models plus a 5-fold CV comparison).
- The two **distance-based** models share one stratified 15k subsample, for two different reasons:
  a full RBF-SVM is O(n²) and impractical on 144k rows, while k-NN is instant to fit but a lazy
  learner's "model" *is* its training set — deploying one trained on 144k rows would mean shipping
  144k patient records inside the artifact. Both are documented as explicit engineering trade-offs,
  and `§6.4b` excludes both from deployment on exactly that ground.
