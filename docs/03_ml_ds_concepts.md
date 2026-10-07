# 03 - ML and data-science concepts for this project (and around it)

Each concept: what it is, why it matters, and where it appears in this project.

---

## Part A - Problem framing

### Supervised learning and classification
You have inputs **X** (features) and a known answer **y** (label). The model learns the mapping from labelled examples. *Classification* predicts a category; here binary: hazardous (1) or not (0). *Regression* would predict a number.

### Features, target, rows
A **feature** is an input column; the **target** is what you predict. Here: features H, velocity, miss distance (and engineered ones); target `hazardous`.

### Probabilities vs hard decisions
Most classifiers output a probability P(y=1). A **decision threshold** turns it into a yes/no. The default 0.5 is arbitrary; choose it from the costs of mistakes.

### Baseline
A simple reference model (logistic regression, or "always predict the majority class"). If a complex model barely beats it, the complexity isn't earning its keep. Here logistic regression came within ~0.01 PR-AUC of XGBoost.

---

## Part B - Data understanding and cleaning

### EDA (exploratory data analysis)
Looking at distributions, relationships, missing values, class balance *before* modelling. It found: repeated asteroids, constant columns, diameter = f(H), strong H effect.

### Data types and parsing
Convert text booleans ("True") to 0/1, strings to numbers. `errors="coerce"` turns unparsable values into NaN instead of crashing.

### Duplicates and constant columns
Exact duplicate rows over-weight the same example. Constant columns have zero variance and carry no information.

### Missing values
Types: MCAR (random), MAR (depends on other columns), MNAR (missingness itself informative). Options: drop rows (loses data), impute, or add a **missing-indicator** column so the model can learn "this was missing". **Median imputation** is robust to outliers (mean is not). Imputation statistics must be learned from training data only (see leakage). This dataset had no missing values, but the pipeline is built for them.

### Outliers and skew
Heavy right-skewed variables (speed, distance, size) are tamed with a **log transform**: it compresses large values, makes relationships more linear, and helps linear models. Tree models don't need it but don't mind.

### Units and domain knowledge
Converting km to lunar distances, km/h to km/s, and using the astronomy formula for diameter are *domain-driven* feature choices; often worth more than a fancier model.

---

## Part C - Astronomy background (domain)
- **Absolute magnitude (H):** brightness of an asteroid if it were 1 AU from both Sun and observer. Lower H = brighter = larger (given similar reflectivity). Each +5 in H is 100x fainter, i.e. 10x smaller in diameter.
- **Albedo:** fraction of light reflected. Size from H depends on it: `D = 1329/sqrt(albedo) * 10^(-H/5)` km.
- **AU:** astronomical unit, Earth-Sun distance (~150 million km). 0.05 AU ~ 7.5 million km.
- **Lunar distance (LD):** Earth-Moon distance, 384,400 km.
- **MOID (Minimum Orbit Intersection Distance):** the smallest distance between the asteroid's *orbit* and Earth's *orbit*. Not the same as the miss distance of one flyby.
- **Orbital elements:** a (semi-major axis), e (eccentricity), i (inclination), plus node, argument of perihelion, mean anomaly. They define the orbit and determine MOID.
- **PHA:** MOID <= 0.05 AU and H <= 22 (> ~140 m).

---

## Part D - Splitting data and avoiding leakage

### Train / validation / test
- **Train:** fit the model. **Validation (CV folds):** choose models and hyperparameters. **Test:** touched once at the end for an honest estimate. Using the test set for decisions makes the final score optimistic.

### Stratified split
Keeps the class ratio (~9.7% positives) in each split. Essential for imbalanced data, else a fold could have too few positives.

### Grouped split (GroupKFold)
When several rows belong to one entity (an asteroid), keep all its rows on the same side. Otherwise the model is tested on things it has effectively memorised. `StratifiedGroupKFold` does both.

### Data leakage
Information in training that wouldn't be available at prediction time, or information from validation/test seeping into training. Types seen here:
1. **Target leakage:** a feature that encodes the label (MOID and H define PHA). Gives suspiciously perfect scores.
2. **Preprocessing leakage:** fitting scaler/imputer/SMOTE on all data before splitting.
3. **Group leakage:** same entity in train and test.
4. **Threshold/hyperparameter tuning on the test set.**
Red flag: a score that is "too good". Always ask "would this feature exist at prediction time?"

### Pipelines
A chain of steps (transform -> transform -> model) treated as one estimator. In cross-validation, each fold re-fits every step on its training part only, which prevents preprocessing leakage and makes deployment simple (one object to save).

### Reproducibility
Fixed random seeds, pinned library versions, saved config.

---

## Part E - Feature engineering and scaling
- **Feature engineering:** creating informative variables from raw ones (logs, ratios, flags, physics-inspired combinations like `log(D^3 v^2)`).
- **Binary flags / thresholds:** `within_moon_orbit`, `size_flag`.
- **Standardisation (z-score):** (x - mean)/std; needed for distance- or gradient-based models (logistic regression, SVM, kNN, neural nets); not needed for trees.
- **Normalisation (min-max):** rescale to [0, 1].
- **Multicollinearity:** highly correlated features (e.g. `miss_distance_ld` and `log_miss_distance`). Harmless for trees; for linear models it makes coefficients unstable. Check with a correlation heatmap.
- **Feature selection:** dropping useless/redundant variables (we dropped the diameter columns).

---

## Part F - Class imbalance
Imbalance = classes of very unequal size. Consequences: accuracy misleads; models favour the majority class.

| Technique | Idea | Notes |
|---|---|---|
| **Class weights** | Penalise mistakes on the minority class more | Cheap, no new data. Won here. |
| **Oversampling (random)** | Duplicate minority rows | Risk of overfitting. |
| **SMOTE** | Create *synthetic* minority points by interpolating between neighbours | Must be applied **inside the training fold only**; never to validation/test data. Can create unrealistic points in noisy data. |
| **Undersampling** | Drop majority rows | Loses information. |
| **Threshold moving** | Change the cut-off instead of the data | Used here. |
| **Better metrics** | PR-AUC, recall, F1 | Used here. |

---

## Part G - Models

### Logistic regression
Linear model: `P(y=1) = sigmoid(w.x + b)`. Fast, interpretable (coefficients), needs scaling; limited to roughly linear boundaries. Regularisation strength `C` (smaller = stronger).

### Decision trees and Random Forest
A tree splits data by feature thresholds. A **random forest** averages many trees trained on bootstrap samples with random feature subsets (**bagging**): lower variance, robust, little tuning. Key hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`. Probabilities are averages of tree votes, so can be poorly calibrated.

### Gradient boosting (XGBoost, LightGBM)
Trees are built **sequentially**, each correcting the previous ones' errors, using gradients of a loss. Very strong on tabular data. Key hyperparameters: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `min_child_weight`, `scale_pos_weight` (class imbalance). `tree_method="hist"` bins feature values for speed.

### Bias-variance trade-off
Too simple => underfit (high bias); too flexible => overfit (high variance). Regularisation, depth limits, more data and CV help balance this.

### Overfitting vs underfitting
Overfit: great train score, worse validation score. Detect with CV or a train/validation gap.

### Other model families (around the project)
SVM, k-NN, naive Bayes, neural networks, LightGBM/CatBoost. For small tabular data, gradient boosting is usually the first strong choice.

---

## Part H - Evaluation metrics

Confusion matrix (positive = hazardous):

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | TN | FP (false alarm) |
| Actual 1 | FN (miss) | TP |

- **Accuracy** = (TP+TN)/all. Misleading with imbalance.
- **Precision** = TP/(TP+FP): of the flagged, how many were real. 
- **Recall (sensitivity, TPR)** = TP/(TP+FN): of the real ones, how many we caught.
- **F1** = harmonic mean of precision and recall.
- **ROC curve:** TPR vs FPR across thresholds. **ROC-AUC:** probability that a random positive is ranked above a random negative; 0.5 = chance. Can look good on imbalanced data because FPR divides by the huge negative class.
- **Precision-recall curve / PR-AUC (average precision):** precision vs recall across thresholds. Baseline = prevalence (~0.096 here). More informative for rare positives. **Headline metric** in this project.
- **Specificity** = TN/(TN+FP).
- **Calibration:** do predicted probabilities match real frequencies (when it says 30%, is it right 30% of the time)? Check with a calibration curve; fix with Platt/isotonic calibration.
- **Cost-sensitive thinking:** choose the operating point from the cost of FN vs FP. Here: recall >= 90%.

### Threshold tuning
Take model probabilities, scan thresholds on the PR curve, pick the one meeting the business rule. Do it on **out-of-fold** predictions, not on test. Trade-off seen in results: at threshold 0.5 recall was 0.975 with precision 0.308; at 0.665 recall 0.907 with precision 0.312.

---

## Part I - Validation and tuning
- **K-fold cross-validation:** split train into K parts; train on K-1, validate on 1; repeat and average. Gives a more stable estimate and a standard deviation.
- **Out-of-fold (OOF) predictions:** each row predicted by a model that did not see it; used for honest threshold selection (`cross_val_predict`).
- **Hyperparameters vs parameters:** hyperparameters are set before training (depth, learning rate); parameters are learned (weights, split points).
- **Grid search vs random search:** grid tries all combinations (explodes); **RandomizedSearchCV** samples N combinations from distributions and is usually just as good for far less time.
- **Nested CV:** an outer loop for evaluation around an inner loop for tuning; more rigorous. Here a held-out test set plays the outer role.
- **Statistical caution:** CV std shows differences of ~0.01 PR-AUC are within noise.

---

## Part J - Interpretability and error analysis
- **Feature importance (model-native):** how much splits on a feature reduce the loss; biased toward high-cardinality/continuous features; says nothing about direction.
- **SHAP (SHapley Additive exPlanations):** from game theory. Each feature's contribution to moving a prediction away from the base (average) value; contributions add up exactly. **Summary plot:** global view (impact vs feature value). **Force plot:** one prediction broken down. TreeExplainer is exact and fast for tree models. For XGBoost the values are in **log-odds** units.
- **Local vs global explanations:** one prediction vs the whole model.
- **Error analysis:** inspect TP/FP/FN/TN groups, slice performance by feature bins to find where the model fails and why (here: large asteroids on slow, distant flybys get missed; big non-hazardous asteroids cause false alarms).
- **Correlation vs causation:** a feature that predicts is not necessarily a cause.

---

## Part K - Engineering and deployment
- **Serialisation (joblib):** save the fitted pipeline to disk and reload it. Save the threshold with it. Loading requires the same library versions and importable custom functions (hence pinned requirements and `select_columns` in `features.py`).
- **Streamlit:** turns a Python script into a web app; re-runs the script on each interaction; `@st.cache_*` avoids reloading heavy objects.
- **Testing ML code:** unit tests for deterministic functions (cleaning, features); smoke tests (model loads and outputs a valid probability); sanity/behavioural tests (bigger asteroid should score higher).
- **Project hygiene:** modular `src/`, config file, `.gitignore`, pinned requirements, Makefile for repeatable commands, README with results.
- **Training/serving skew:** if the app computes features differently than training you get silent errors. Solved by putting feature engineering inside the pipeline.
- **Model monitoring and drift (beyond this project):** in production, watch input distributions and performance over time; retrain when they drift.

---

## Part L - Limitations and what to learn next
- **Information ceiling:** no model can recover information absent from the data (MOID). Better algorithms don't fix that; better features do.
- **Label definition:** a label created by a formula is usually better *computed* than predicted; ML adds value only when the formula's inputs are unavailable, which is exactly this project's situation.
- **Next steps:** use JPL orbital elements; calibrate probabilities; one-row-per-asteroid aggregation; try LightGBM/CatBoost; cost-based threshold selection; an LLM layer turning SHAP values into plain-English explanations; model cards documenting intended use and limits.

---

## Glossary (quick)
| Term | One-liner |
|---|---|
| Prevalence | fraction of positives |
| OOF | out-of-fold prediction |
| FP / FN | false alarm / miss |
| AUC | area under a curve; summarises ranking quality |
| Hyperparameter | a setting you choose before training |
| Regularisation | penalty that discourages overly complex models |
| Ensemble | combining many models (bagging, boosting) |
| Log-odds | log(p/(1-p)); the scale logistic models and XGBoost SHAP use |
| Pipeline | chained preprocessing + model treated as one estimator |
