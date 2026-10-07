# 01 - Project workflow: the idea, the problem, the "why"

## 1. The idea in one paragraph
Near-Earth asteroids are tracked by NASA. A small subset is labelled **Potentially Hazardous Asteroids (PHAs)**: rocks big enough and on orbits close enough to Earth's that they deserve extra monitoring. This project trains a machine-learning classifier that looks at the numbers describing an asteroid's close approach and predicts "hazardous / not hazardous", then explains *why* it said so and serves the result in a small web app.

## 2. What problem does it solve?
- **Practical framing:** astronomers must decide which of tens of thousands of objects deserve follow-up observations. A model that ranks objects by hazard probability helps prioritise limited telescope time.
- **Learning framing (the real value for a portfolio):** the project demonstrates the full data-science lifecycle on a realistic, *messy* problem: imbalanced classes, repeated rows for the same object, a label that is defined by variables you may not have, and a metric choice where "accuracy" is misleading.

## 3. Why is it hard? (the interesting parts)
| Difficulty | What it means here | How the project handles it |
|---|---|---|
| **Class imbalance** | Only ~9.7% of rows are hazardous. A model that always says "not hazardous" is 90% accurate and useless. | PR-AUC / recall as headline metrics, class weights vs SMOTE comparison, tuned threshold. |
| **Cost asymmetry** | Missing a real hazard is much worse than a false alarm. | Decision threshold chosen to reach >= 90% recall. |
| **Grouped data** | 90,836 rows but only 27,423 asteroids: the same asteroid appears many times (once per close approach). | Splits and CV are grouped by asteroid so no asteroid is in both train and test. |
| **Label leakage** | PHA is *defined* by MOID <= 0.05 AU and H <= 22. Any feature that reproduces the definition makes the task trivial. | Three experiments (full / flyby-only / H-only) measure how much each piece contributes. |
| **Missing the key variable** | The dataset has no MOID or orbital elements. | Honest reporting: precision is capped near 30%; documented as a limitation and future work. |

## 4. The dataset
Kaggle "NASA - Nearest Earth Objects" (`neo.csv`). One row = one close approach.

| Column | Meaning |
|---|---|
| `name`, `id` | asteroid identity (repeats across rows) |
| `est_diameter_min/max` | size estimate in km, computed from H (redundant with H) |
| `relative_velocity` | speed relative to Earth, km/h |
| `miss_distance` | distance at closest approach, km |
| `absolute_magnitude` (H) | intrinsic brightness; lower = brighter = bigger |
| `orbiting_body`, `sentry_object` | constant (dropped) |
| `hazardous` | the label (True/False) |

## 5. End-to-end workflow (what happens, in order)

```
data/raw/neo.csv
   |  src/fetch_data.py   -> checks file exists, prints Kaggle instructions if not
   v
src/clean.py              -> parse types, hazardous -> 0/1, drop duplicates & constants,
   |                         invalid values -> NaN, add `asteroid` group key, report balance
   v
data/processed/neo_clean.csv
   |
   v
src/train.py
   1. split_data():   stratified + grouped 80/20 train/test split (seed 42)
   2. compare:        3 models x 2 imbalance strategies, 5-fold grouped stratified CV
   3. pick best by CV PR-AUC  -> XGBoost + class weights
   4. tune:           RandomizedSearchCV (15 iterations) on PR-AUC
   5. threshold:      out-of-fold predictions on TRAIN -> highest threshold with recall >= 0.90
   6. final fit for 3 experiments (A full, B flyby-only, C H-only), score on untouched TEST
   7. save:           models/pha_model.joblib (pipeline + threshold), reports/metrics.json
   v
src/evaluate.py           -> confusion matrix, ROC, PR, feature importance, SHAP summary/force
   |                         plots, error analysis (reports/figures, reports/error_analysis.md)
   v
app/streamlit_app.py      -> user enters H, speed, miss distance -> probability + verdict
                             + SHAP explanation + dataset scatter + About
```
Alongside: `notebooks/01_eda.ipynb` (exploration before modelling), `tests/` (pytest), `Makefile`, `requirements.txt`.

## 6. Key design decisions and the reasoning

1. **Group by asteroid when splitting.** If asteroid X's approach #1 is in train and approach #2 in test, the model can "recognise" X by its H value (H is constant per asteroid) and get an inflated score. Grouping removes that cheat. This is the most important correctness decision in the project.
2. **Everything inside one Pipeline.** Feature engineering, imputation, scaling and SMOTE are pipeline steps. During cross-validation each fold re-fits them on that fold's training part only, so no information from the validation fold leaks in. Doing SMOTE or scaling *before* splitting is a classic beginner leak.
3. **PR-AUC as the headline metric.** With ~10% positives, ROC-AUC looks flattering because the huge negative class makes false-positive *rates* tiny. PR-AUC focuses on how well you find positives and how pure the flagged set is. Its chance level equals the prevalence (~0.096), so 0.474 is about 5x chance.
4. **Threshold tuned on out-of-fold training predictions, not on test.** Choosing it on the test set would be tuning on your exam. Out-of-fold predictions are honest (each row predicted by a model that never saw it).
5. **Leakage experiment re-framed.** The brief assumed MOID was available. It wasn't, so instead of pretending, the experiments measure what H alone, flyby alone and both together give. Result: H alone = ROC-AUC 0.879; flyby alone = 0.691; both = 0.903.
6. **Pipeline stores threshold with the model.** `models/pha_model.joblib` holds a dict (`pipeline`, `threshold`, `features`, ...), so the app uses exactly what training decided.

## 7. What the results say
- Main model: test PR-AUC 0.474, ROC-AUC 0.903, recall 0.907, precision 0.312 at threshold 0.665.
- Interpretation: it catches ~91% of hazardous rows, but about 2 of every 3 flagged rows are false alarms. The reason is physical: H tells you "big enough", but the other half of the definition (how close the *orbit* gets) isn't in the data.
- Even the simple logistic regression baseline is within ~0.01 PR-AUC of XGBoost, so the limitation is information, not model power.

## 8. How to talk about this in an interview
- "I discovered rows were repeated per asteroid and used grouped stratified splits to prevent leakage."
- "I chose PR-AUC and a recall-targeted threshold because of 10% prevalence and asymmetric costs."
- "I tested what happens when you remove the defining variable and quantified how much signal remained, rather than reporting an inflated score."
- "The honest ceiling is ~30% precision at 90% recall because MOID isn't available; next step is JPL orbital elements."
