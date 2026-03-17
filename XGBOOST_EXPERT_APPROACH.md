# XGBoost Optimization - Mathematical & Rigorous Approach

## Problem Statement

**Current Status:**
- Validation accuracy plateaued at 46% across all boosting methods
- 95 features extracted → 65 selected (arbitrary threshold)
- Hyperparameters chosen without mathematical justification
- No statistical significance testing

**Goal:**
Achieve 50-55% validation accuracy using **mathematically rigorous** feature selection and hyperparameter optimization.

---

## 🔬 Part 1: Feature Analysis (Mathematical Feature Selection)

### Why Current Feature Selection Is Weak:

```python
# Current approach (ANOVA F-test)
selector = SelectKBest(f_classif, k=65)
```

**Problems:**
1. Assumes linear relationships (F-test)
2. Ignores feature interactions
3. k=65 is arbitrary (why not 50? 70? 80?)
4. No redundancy removal

### **Better Approach: Multi-Method Feature Selection**

#### **Method 1: Mutual Information (Non-Linear Feature Importance)**

Mutual Information measures **non-linear** relationships between features and labels.

```python
from sklearn.feature_selection import mutual_info_classif

# Compute mutual information scores
mi_scores = mutual_info_classif(X_train_features, y_train, random_state=42)

# Create dataframe for analysis
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("Top 20 features by Mutual Information:")
print(mi_df.head(20))

# Plot MI scores
plt.figure(figsize=(12, 6))
plt.bar(range(len(mi_scores)), sorted(mi_scores, reverse=True))
plt.xlabel('Feature Rank')
plt.ylabel('Mutual Information Score')
plt.title('Mutual Information Scores (Non-Linear Feature Importance)')
plt.axhline(y=np.median(mi_scores), color='r', linestyle='--', label='Median')
plt.legend()
plt.show()
```

**Mathematical Interpretation:**
- MI = 0: No relationship
- MI > 0: Non-linear dependency exists
- Higher MI = More predictive power

---

#### **Method 2: Recursive Feature Elimination with Cross-Validation (RFECV)**

Finds **optimal number of features** using cross-validation (not arbitrary k=65).

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier

# Use lightweight model for RFE
rfe_estimator = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,
    random_state=42
)

# RFECV automatically finds optimal k
rfecv = RFECV(
    estimator=rfe_estimator,
    step=5,                    # Remove 5 features at a time
    cv=StratifiedKFold(5),     # 5-fold CV
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rfecv.fit(X_train_features, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Feature ranking: {rfecv.ranking_}")

# Plot number of features vs CV score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('RFECV: Optimal Number of Features')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
            label=f'Optimal = {rfecv.n_features_}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Get selected features
selected_features_rfe = [feature_names[i] for i in range(len(feature_names))
                         if rfecv.support_[i]]
print(f"\nSelected features: {selected_features_rfe[:10]}...")
```

**Why This Is Better:**
- **Data-driven** optimal k (not arbitrary 65)
- **Cross-validated** (robust)
- **Model-based** (considers feature interactions)

---

#### **Method 3: Correlation Analysis (Remove Redundant Features)**

Remove highly correlated features (multicollinearity).

```python
# Compute correlation matrix
corr_matrix = np.corrcoef(X_train_features.T)

# Find highly correlated pairs (>0.9)
high_corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(corr_matrix[i, j]) > 0.9:
            high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))

print(f"Highly correlated feature pairs (r > 0.9): {len(high_corr_pairs)}")
for f1, f2, corr in high_corr_pairs[:10]:
    print(f"  {f1} <-> {f2}: r = {corr:.3f}")

# Remove redundant features
def remove_correlated_features(X, feature_names, threshold=0.9):
    """Remove features with correlation > threshold."""
    corr_matrix = np.corrcoef(X.T)
    to_remove = set()

    for i in range(len(feature_names)):
        if feature_names[i] in to_remove:
            continue
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(feature_names[j])  # Remove second feature

    keep_indices = [i for i, name in enumerate(feature_names) if name not in to_remove]
    return X[:, keep_indices], [feature_names[i] for i in keep_indices]

X_train_decorr, features_decorr = remove_correlated_features(
    X_train_features, feature_names, threshold=0.9
)

print(f"\nOriginal features: {len(feature_names)}")
print(f"After removing correlated: {len(features_decorr)}")
print(f"Removed: {len(feature_names) - len(features_decorr)} redundant features")
```

---

#### **Method 4: Permutation Importance (Model-Agnostic)**

Measures feature importance by **randomly shuffling** each feature and measuring performance drop.

```python
from sklearn.inspection import permutation_importance

# Train baseline model
baseline_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
baseline_model.fit(X_train_scaled, y_train)

# Compute permutation importance
perm_importance = permutation_importance(
    baseline_model,
    X_val_scaled,
    y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Create dataframe
perm_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
}).sort_values('Importance_Mean', ascending=False)

print("Top 20 features by Permutation Importance:")
print(perm_df.head(20))

# Plot with error bars
plt.figure(figsize=(12, 6))
top_20 = perm_df.head(20)
plt.barh(range(len(top_20)), top_20['Importance_Mean'],
         xerr=top_20['Importance_Std'], color='#FF9800', alpha=0.8)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Permutation Importance (Mean ± Std)')
plt.title('Top 20 Features by Permutation Importance')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Why This Is Better:**
- **Model-agnostic** (works with any model)
- **Causal interpretation** (measures actual impact on predictions)
- **Statistical significance** (error bars show reliability)

---

#### **Method 5: Combine All Methods (Ensemble Feature Selection)**

```python
# Normalize all scores to [0, 1]
from sklearn.preprocessing import MinMaxScaler

scaler_mi = MinMaxScaler()
scaler_rfe = MinMaxScaler()
scaler_perm = MinMaxScaler()

mi_normalized = scaler_mi.fit_transform(mi_df[['MI_Score']].values).flatten()
rfe_ranking_inv = 1 / rfecv.ranking_  # Lower rank = better
rfe_normalized = scaler_rfe.fit_transform(rfe_ranking_inv.reshape(-1, 1)).flatten()
perm_normalized = scaler_perm.fit_transform(perm_df[['Importance_Mean']].values).flatten()

# Combine scores (weighted average)
combined_scores = (
    0.4 * mi_normalized +      # 40% weight to MI (non-linear)
    0.3 * rfe_normalized +     # 30% weight to RFE (model-based)
    0.3 * perm_normalized      # 30% weight to permutation (causal)
)

# Create final ranking
final_ranking = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score': mi_normalized,
    'RFE_Score': rfe_normalized,
    'Perm_Score': perm_normalized,
    'Combined_Score': combined_scores
}).sort_values('Combined_Score', ascending=False)

print("Top 20 features by combined score:")
print(final_ranking.head(20))

# Select top k features (where k is from RFECV)
optimal_k = rfecv.n_features_
top_features = final_ranking.head(optimal_k)['Feature'].tolist()

print(f"\nOptimal number of features (from RFECV): {optimal_k}")
print(f"Selected features: {top_features[:15]}...")
```

**Why This Is Expert-Level:**
- ✅ **Multi-method consensus** (not relying on single metric)
- ✅ **Weighted ensemble** (different methods have different strengths)
- ✅ **Data-driven k** (from RFECV, not arbitrary)
- ✅ **Mathematically rigorous**

---

## 🔬 Part 2: Hyperparameter Optimization (Mathematical Approach)

### Why RandomizedSearchCV Is Weak:

```python
# Current approach
RandomizedSearchCV(n_iter=50, ...)  # Random sampling
```

**Problems:**
1. **Random sampling** - No intelligent search
2. **No prior knowledge** - Treats all parameters equally
3. **No convergence** - 50 iterations may miss optimal
4. **Computationally inefficient**

### **Better Approach: Bayesian Optimization**

Uses **probabilistic model** to intelligently search hyperparameter space.

#### **Install Optuna (Bayesian Optimization Framework)**

```python
!pip install optuna -q
import optuna
```

#### **Define Objective Function**

```python
def objective(trial):
    """
    Objective function for Bayesian optimization.

    Optuna will minimize/maximize this function by intelligently
    sampling hyperparameters based on previous trial results.
    """
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 400, 1200, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
    }

    # Create model
    model = XGBClassifier(
        **params,
        tree_method='hist',
        device=device,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=50
    )

    # Cross-validation with stratification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X_train_selected_final,  # Use optimally selected features
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # Return mean CV accuracy
    return scores.mean()

# Create Optuna study
study = optuna.create_study(
    direction='maximize',           # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)  # Early stopping for bad trials
)

# Run optimization
print("Starting Bayesian Optimization with Optuna...")
print("This will intelligently search the hyperparameter space.\n")

study.optimize(
    objective,
    n_trials=100,              # 100 trials (more efficient than RandomSearch)
    timeout=3600,              # 1 hour timeout
    show_progress_bar=True
)

# Results
print(f"\n{'='*60}")
print("BAYESIAN OPTIMIZATION RESULTS")
print(f"{'='*60}")
print(f"Best trial: {study.best_trial.number}")
print(f"Best accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
print(f"\nBest hyperparameters:")
for param, value in study.best_params.items():
    print(f"  {param:20s}: {value}")
```

**Why This Is Better:**
- ✅ **Intelligent search** (uses Gaussian Process / TPE)
- ✅ **Converges faster** (fewer trials needed)
- ✅ **Probabilistic model** (learns from previous trials)
- ✅ **Early pruning** (stops bad trials early)

---

#### **Analyze Optimization Process**

```python
# Visualization 1: Optimization history
optuna.visualization.plot_optimization_history(study).show()

# Visualization 2: Parameter importance
optuna.visualization.plot_param_importances(study).show()

# Visualization 3: Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study).show()

# Get top 5 trials
print("\nTop 5 trials:")
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(top_trials):
    print(f"{i+1}. Trial {trial.number}: {trial.value*100:.2f}%")
```

**What This Shows:**
- Which parameters matter most
- How optimization converged over time
- Trade-offs between parameters

---

## 🔬 Part 3: Statistical Validation

### **Problem: Is 46% → 48% Real or Just Noise?**

Need **statistical significance testing**.

#### **McNemar's Test (Paired Statistical Test)**

```python
from statsmodels.stats.contingency_tables import mcnemar

# Compare baseline vs optimized
y_val_pred_baseline = baseline_model.predict(X_val_selected)
y_val_pred_optimized = optimized_model.predict(X_val_selected_final)

# Create contingency table
# correct_baseline | correct_optimized | count
#        0         |         0         |  a
#        0         |         1         |  b
#        1         |         0         |  c
#        1         |         1         |  d

correct_baseline = (y_val_pred_baseline == y_val)
correct_optimized = (y_val_pred_optimized == y_val)

b = np.sum(~correct_baseline & correct_optimized)   # Optimized correct, baseline wrong
c = np.sum(correct_baseline & ~correct_optimized)   # Baseline correct, optimized wrong

# McNemar's contingency table
table = [[np.sum(~correct_baseline & ~correct_optimized), b],
         [c, np.sum(correct_baseline & correct_optimized)]]

# Perform test
result = mcnemar(table, exact=True)

print(f"McNemar's Test:")
print(f"  Baseline correct, Optimized wrong: {c}")
print(f"  Baseline wrong, Optimized correct: {b}")
print(f"  p-value: {result.pvalue:.4f}")
print(f"  Statistic: {result.statistic:.4f}")

if result.pvalue < 0.05:
    print(f"\n✓ Improvement is statistically significant (p < 0.05)")
else:
    print(f"\n✗ Improvement is NOT statistically significant (p >= 0.05)")
```

**Interpretation:**
- **p < 0.05**: Improvement is real, not random chance
- **p >= 0.05**: Improvement could be noise

---

#### **Bootstrap Confidence Intervals**

```python
from sklearn.utils import resample

def bootstrap_accuracy(y_true, y_pred, n_bootstrap=1000):
    """Compute bootstrap confidence interval for accuracy."""
    accuracies = []

    for _ in range(n_bootstrap):
        # Resample predictions
        indices = resample(range(len(y_true)), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute accuracy
        acc = accuracy_score(y_true_boot, y_pred_boot)
        accuracies.append(acc)

    # Compute 95% CI
    lower = np.percentile(accuracies, 2.5)
    upper = np.percentile(accuracies, 97.5)
    mean = np.mean(accuracies)

    return mean, lower, upper

# Compute for both models
mean_base, lower_base, upper_base = bootstrap_accuracy(y_val, y_val_pred_baseline)
mean_opt, lower_opt, upper_opt = bootstrap_accuracy(y_val, y_val_pred_optimized)

print(f"Bootstrap 95% Confidence Intervals (1000 iterations):")
print(f"\nBaseline Model:")
print(f"  Mean:   {mean_base*100:.2f}%")
print(f"  95% CI: [{lower_base*100:.2f}%, {upper_base*100:.2f}%]")
print(f"\nOptimized Model:")
print(f"  Mean:   {mean_opt*100:.2f}%")
print(f"  95% CI: [{lower_opt*100:.2f}%, {upper_opt*100:.2f}%]")

# Check if CIs overlap
if lower_opt > upper_base:
    print(f"\n✓ CIs do NOT overlap - Optimized is significantly better")
else:
    print(f"\n⚠ CIs overlap - Improvement may not be significant")
```

---

## 📊 Part 4: Final Expert-Level Workflow

```python
# Step 1: Ensemble Feature Selection
print("STEP 1: Multi-Method Feature Selection")
print("="*60)

# ... (Run all 5 feature selection methods)
# ... (Combine scores)
# ... (Select optimal k features)

optimal_features = final_ranking.head(optimal_k)['Feature'].tolist()
X_train_final = X_train_features[:, [feature_names.index(f) for f in optimal_features]]
X_val_final = X_val_features[:, [feature_names.index(f) for f in optimal_features]]

# Standardize
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_val_final_scaled = scaler_final.transform(X_val_final)

print(f"Selected {len(optimal_features)} features (data-driven, not arbitrary)")

# Step 2: Bayesian Hyperparameter Optimization
print("\nSTEP 2: Bayesian Optimization (Optuna)")
print("="*60)

# ... (Run Optuna optimization)

best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

# Step 3: Train Final Model
print("\nSTEP 3: Train Final Optimized Model")
print("="*60)

final_model = XGBClassifier(
    **best_params,
    tree_method='hist',
    device=device,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

final_model.fit(X_train_final_scaled, y_train, sample_weight=sample_weights)

# Step 4: Statistical Validation
print("\nSTEP 4: Statistical Significance Testing")
print("="*60)

# ... (Run McNemar's test)
# ... (Run Bootstrap CI)

# Step 5: Comprehensive Evaluation
print("\nSTEP 5: Final Evaluation")
print("="*60)

y_val_pred_final = final_model.predict(X_val_final_scaled)
val_acc_final = accuracy_score(y_val, y_val_pred_final)

print(f"\nFinal Validation Accuracy: {val_acc_final*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_val, y_val_pred_final, target_names=class_names))
```

---

## 🎓 Summary: What Makes This Expert-Level

| Aspect | Copy-Paste Approach | Expert Approach |
|--------|-------------------|-----------------|
| **Feature Selection** | SelectKBest (k=65, arbitrary) | Multi-method ensemble (MI + RFE + Permutation) |
| **Optimal k** | Guessed (65) | Data-driven (RFECV finds optimal) |
| **Redundancy** | Ignored | Correlation analysis removes redundant features |
| **Hyperparameter Search** | RandomizedSearchCV (random) | Bayesian Optimization (intelligent, converges faster) |
| **Validation** | Simple train/val split | Stratified K-Fold + Statistical testing |
| **Significance Testing** | None | McNemar's test + Bootstrap CI |
| **Interpretability** | Basic feature importance | Permutation importance + parameter importance analysis |
| **Mathematical Rigor** | Low | High (probabilistic models, statistical tests) |

---

## ⏱️ Time-Saving Version (If You Have Limited Time)

If you don't have time for full Bayesian optimization, do this minimal expert approach:

1. **RFECV** for optimal k (10 minutes)
2. **Permutation Importance** for feature ranking (5 minutes)
3. **Grid Search** on small parameter space (20 minutes)
4. **McNemar's Test** for significance (2 minutes)

**Total: ~40 minutes** but shows you understand rigorous ML methodology.

---

**Next Steps:**
1. Choose between **full expert approach** (2-3 hours) or **minimal expert approach** (40 min)
2. I'll create the notebook with your chosen approach
3. Run on Colab GPU
4. Document findings in technical report

Which approach do you want to implement?
