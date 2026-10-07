# 02 - Code walkthrough: every file explained

Folder map:
```
src/config.py  src/fetch_data.py  src/clean.py  src/features.py  src/train.py  src/evaluate.py
app/streamlit_app.py   tests/test_clean.py  tests/test_model_smoke.py
notebooks/01_eda.ipynb   Makefile   requirements.txt   .gitignore
```
Modules inside `src/` import each other with `from src.x import ...`, so run them as modules from the project root (`python -m src.train`), not as `python src/train.py`.

---

## src/config.py - one place for every constant
No logic, only values, so nothing is hard-coded elsewhere.
- `ROOT = Path(__file__).resolve().parents[1]`: the project folder, found from the file's own location (works from any working directory).
- Path constants: `RAW_CSV`, `CLEAN_CSV`, `MODEL_PATH`, `METRICS_PATH`, `REPORTS_DIR`, `FIG_DIR`.
- `SEED = 42`: used by the split, CV, SMOTE, models and search so results reproduce.
- `TEST_SIZE = 0.20`, `N_FOLDS = 5`.
- `TARGET = "hazardous"`; `GROUP = "asteroid"`: the column used to keep an asteroid's rows together.
- `TARGET_RECALL = 0.90`: the recall the threshold must reach.
- `RAW_FEATURES`: the 3 raw inputs the model needs (H, velocity, miss distance).
- Feature groups: `SIZE_FEATURES` (H-based), `FLYBY_FEATURES` (speed/distance-based).
- `EXPERIMENTS`: a dict mapping experiment name -> list of allowed engineered features:
  - `A_full` = size + flyby (main), `B_flyby_only`, `C_H_only`.
- `MAIN_EXPERIMENT = "A_full"`.

---

## src/fetch_data.py - locate the dataset
- `INSTRUCTIONS`: an f-string with Kaggle download steps, including the exact destination path.
- `check_raw_data() -> bool`: if `RAW_CSV` exists, prints its size and returns True; otherwise prints instructions and returns False.
- `if __name__ == "__main__": sys.exit(0 if check_raw_data() else 1)`: exit code 0/1 so `make fetch` fails visibly if data is missing.
Why not auto-download? Kaggle requires login/API credentials; a clear instruction is more robust than a fragile script.

---

## src/clean.py - turn raw CSV into a trustworthy table
Constants: `NUMERIC_COLS` columns that must be numbers.

- `load_raw(path)`: `pd.read_csv`.
- `parse_target(series)`: lowercases strings and maps `true/t/y/yes/1 -> 1`, `false/f/n/no/0 -> 0`. Handles real booleans too (they become the strings "true"/"false"). If anything fails to map (NaN after mapping) it raises `ValueError`. Returns ints.
- `drop_constant_columns(df)`: drops columns with <= 1 unique value (here `orbiting_body`, `sentry_object`). They can't help any model.
- `clean(df)`, in order:
  1. copy (never mutate the caller's data);
  2. convert target to 0/1;
  3. `pd.to_numeric(..., errors="coerce")` on numeric columns (bad text -> NaN);
  4. `drop_duplicates()`;
  5. drop constants;
  6. velocity or miss distance `<= 0` -> NaN (physically impossible; the pipeline imputer will handle them);
  7. create `asteroid` = stripped `name`, the grouping key;
  8. `reset_index`.
  Missing values are deliberately **not** filled here: filling with the median of the whole dataset would leak test information. The median imputer lives in the pipeline and is fit on training data only.
- `class_balance(df)`: counts and share (rounded) per class.
- `diameter_h_formula_error(df)`: computes `D = 1329/sqrt(0.25) * 10**(-H/5)` and returns the maximum `|log10(est_diameter_min / D)|`. It returned 0.0000, proving diameters are just H re-expressed.
- `__main__` block: loads, cleans, prints rows/unique asteroids/missing count/dropped columns/class balance, saves `neo_clean.csv`.

---

## src/features.py - feature engineering
Every feature is a pure function of the 3 raw columns, so the *same code* runs in training, in the saved model and in the app.
- `LUNAR_DISTANCE_KM = 384,400`; `H_PHA_LIMIT = 22`.
- `diameter_km_from_h(h, albedo=0.15)`: `1329/sqrt(albedo) * 10**(-H/5)` (standard astronomy formula). Used for the energy proxy and for the app's size caption. Accepts scalars or arrays.
- `select_columns(X, cols)`: returns `X[cols]`. Lives here (not in train.py) so the pickled pipeline can always import it from a stable module path.
- `add_features(X)`: builds a new DataFrame with:
  - `absolute_magnitude`: passthrough.
  - `size_flag`: 1 if H <= 22 (`.where(h.notna())` keeps NaN when H is missing).
  - `velocity_kms`: km/h / 3600.
  - `log_velocity`: log10 speed (reduces right skew).
  - `miss_distance_ld`: km / 384,400.
  - `log_miss_distance`: log10 of that.
  - `within_moon_orbit`, `within_10_ld`: binary flags for very close flybys.
  - `log_energy_proxy`: `log10(D^3 * v^2)`, a kinetic-energy-like quantity (mass ~ size^3).
  Note: the ratio/Earth-crossing features from the original orbital-element brief need `q`, `ad`, `e`; they are not possible here.

---

## src/train.py - the heart of the project

### Imports worth knowing
`imblearn.pipeline.Pipeline` (not sklearn's) because it supports a sampler step like SMOTE; `StratifiedGroupKFold` for grouped + stratified splitting; `scipy.stats` distributions for random search ranges.

### `SCORING`
Dict of metric names -> sklearn scorers: recall, precision, f1, roc_auc, `average_precision` (= PR-AUC).

### Data functions
- `load_data()`: reads `neo_clean.csv`; if absent, cleans raw and saves it first.
- `split_data(df)`: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, takes the **first fold** as test (= 20%). Result: classes stratified, asteroids never split. Returns two DataFrames with reset indexes.
- `cv_splitter()`: a fresh 5-fold `StratifiedGroupKFold` (new object each call so it is not shared state).

### Model/pipeline construction
- `make_classifier(name, imbalance, pos_weight)`:
  - `logreg`: `LogisticRegression(max_iter=2000)`; `class_weight="balanced"` if imbalance is `class_weight`.
  - `random_forest`: 150 trees, `min_samples_leaf=3`, `n_jobs=-1`; `class_weight="balanced_subsample"` when weighted.
  - `xgboost`: 300 trees, depth 5, lr 0.1, `subsample=0.8`, `tree_method="hist"` (fast), `eval_metric="aucpr"`; `scale_pos_weight = negatives/positives` when weighted.
  - For `smote`, no weights are set (SMOTE does the balancing).
- `build_pipeline(name, imbalance, feature_cols, pos_weight)`: assembles steps
  1. `features`: `FunctionTransformer(add_features)`;
  2. `select`: `FunctionTransformer(select_columns, kw_args={"cols": feature_cols})` (this is how each experiment gets its feature subset);
  3. `impute`: `SimpleImputer(strategy="median", add_indicator=True)` (adds "was missing" columns only for features that had NaNs in training; none here, so shape is unchanged);
  4. `scale`: `StandardScaler` (needed by logistic regression, harmless for trees);
  5. optionally `smote`: `SMOTE(random_state=42)`, which only acts during `fit`, never in `predict`;
  6. `clf`.
- `PARAM_GRIDS`: search spaces keyed by model; parameter names use the `clf__` prefix (pipeline step name + double underscore).

### Metrics helpers
- `pick_threshold(y_true, proba, target_recall)`: `precision_recall_curve` returns `precision`, `recall` (each length n+1) and `thresholds` (length n); as the threshold rises recall falls. `recall[:-1] >= target` finds thresholds that still reach the target; the **last** such index is the highest qualifying threshold, which gives the best precision. Fallback: the lowest threshold.
- `score_at(y_true, proba, threshold)`: binarises predictions at the threshold; returns recall, precision, F1, ROC-AUC, PR-AUC (these two use raw probabilities, independent of threshold) and confusion-matrix counts `tn, fp, fn, tp`.
- `fit_and_report(pipe, train, test)`:
  1. `cross_val_predict(..., method="predict_proba")`: out-of-fold probabilities for every training row;
  2. `pick_threshold` on those;
  3. `pipe.fit` on all training data;
  4. score the **test** set at the tuned threshold and at 0.5;
  returns a report dict and the fitted pipeline.

### `main()` step by step
1. Load and split; assert no asteroid appears in both sets (safety net for the grouping).
2. `pos_weight = negatives / positives` from the training set.
3. Loop 3 models x 2 strategies with `cross_validate(..., groups=gtr)`, storing mean and std of each metric; print progress.
4. `best = max(comparison, key=pr_auc)`.
5. `RandomizedSearchCV(n_iter=15, scoring="average_precision", refit=False)`; `search.fit(X, y, groups=g)` passes groups to the CV splitter. `refit=False` because we refit ourselves for each experiment.
6. For each experiment: build the pipeline, `set_params(**best_params)`, `fit_and_report`. Keep the main one.
7. `joblib.dump` a dict (`pipeline`, `threshold`, `features`, `raw_features`, `model_name`, `imbalance`); write `metrics.json` (numpy scalars converted with `.item()`).

---

## src/evaluate.py - figures and error analysis
- `save(fig, name)`: ensures `reports/figures` exists, saves 150-dpi PNG with tight bounding box, closes the figure (avoids memory leaks). `matplotlib.use("Agg")` makes it work without a display.
- `transformed_matrix(pipe, X, names)`: `pipe[:-1].transform(X)` runs all steps except the classifier (samplers are skipped when transforming) and wraps the result in a DataFrame with feature names. SHAP's tree explainer needs the classifier's own input, which is this transformed matrix.
- `main()`:
  1. load bundle, split (same seed => same test set as training), predict probabilities, apply the saved threshold;
  2. confusion matrix, ROC (with chance diagonal), PR curve (with prevalence baseline);
  3. native feature importance (`feature_importances_`, XGBoost's gain-based);
  4. SHAP: 2,000-row random test sample (speed); `shap.TreeExplainer(clf)`; handles both 2-D (XGBoost) and 3-D (Random Forest) outputs; `summary_plot` is drawn with **real, unscaled** feature values (`add_features(sample)[names]`) so colours and labels make sense even though SHAP values were computed on the scaled matrix;
  5. force plots for the most confident true positive and one false negative;
  6. error analysis: label each row TP/FN/FP/TN, median feature values per outcome, performance per H-bin (`pd.cut`), a two-panel figure, a markdown report, and an `error_analysis` section appended to `metrics.json`.

---

## app/streamlit_app.py - the web app
- `sys.path.insert(0, project_root)` so `import src...` works when Streamlit launches the file.
- `@st.cache_resource load_model()`: load the model once and share it; `@st.cache_data load_dataset()`: cache the DataFrame.
- If the model file is missing, show an error and `st.stop()`.
- Sidebar sliders: H (9-33), speed in km/s (0.1-60), miss distance in lunar distances (0.01-195). A caption shows estimated width in metres and km.
- Convert inputs to the model's raw units (km/s x 3600 -> km/h; LD x 384,400 -> km), build a 1-row DataFrame in `RAW_FEATURES` order, `predict_proba`, compare with the saved threshold.
- Tab "Predict": three metrics (probability, threshold, verdict) + SHAP force plot for this single row (transform with `pipe[:-1]`, explain with `TreeExplainer`, show real feature values).
- Tab "Dataset explorer": Plotly scatter of 8,000 sampled rows (H vs miss distance, log y), coloured by class, a dashed line at H = 22, and your input as a gold star.
- Tab "About": PHA definition and model limits.

---

## tests/
- `test_clean.py`: builds a tiny DataFrame (with a duplicate row and a negative velocity) and checks: target parsing variants; invalid target raises; duplicates and constant columns dropped; impossible values become NaN without dropping the row; class shares sum to 1; engineered feature values (e.g. 36,000 km/h = 10 km/s, size flag, 10 LD).
- `test_model_smoke.py`: skipped if no model file. Loads the saved bundle and checks a sample row yields a probability in [0, 1] with a valid threshold, and that a big asteroid (H=19) scores higher than a tiny one (H=30).
Run with `python -m pytest -q`.

---

## notebooks/01_eda.ipynb
Narrated EDA, in order: data structure (rows vs unique asteroids), class balance, distributions by class, proof diameters derive from H, correlation heatmap, H vs miss distance scatter, PHA rate by H bin, takeaways. Executed so outputs are saved.

## Makefile, requirements.txt, .gitignore
- Makefile targets: `fetch`, `clean`, `train`, `evaluate`, `app`, `test`, `all`. `PY ?= python` lets you override the interpreter. Recipes must be indented with tabs.
- `requirements.txt`: pinned versions (`==`) of the exact libraries used for reproducibility.
- `.gitignore`: excludes raw/processed data, the `.joblib` model, caches and virtual environments (large or regenerable files stay out of git).
