# Bank Customer Churn Prediction
# Author: Kevinz Adhi Anggoro | Digital Skola Data Science Bootcamp
# Dataset: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

# ─── INSTALL (jalankan sekali di terminal) ────────────────────────────────────
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap

# ==============================================================================
# 1. IMPORT LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("✅ All libraries imported successfully!")

# ==============================================================================
# 2. LOAD DATASET
# ==============================================================================
# Download dari Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
# Simpan sebagai: Churn_Modelling.csv

df = pd.read_csv("Churn_Modelling.csv")

print(f"\nDataset shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
df.head()

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "="*50)
print("3. EXPLORATORY DATA ANALYSIS")
print("="*50)

# --- 3.1 Basic Info ---
print("\n--- Data Types & Missing Values ---")
print(df.info())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# --- 3.2 Target Distribution ---
print(f"\n--- Churn Distribution ---")
churn_counts = df["Exited"].value_counts()
churn_pct    = df["Exited"].value_counts(normalize=True) * 100
print(pd.DataFrame({"Count": churn_counts, "Percentage (%)": churn_pct.round(1)}))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
axes[0].pie(
    churn_counts,
    labels=["Retained (0)", "Churned (1)"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90
)
axes[0].set_title("Churn Distribution", fontsize=13, fontweight="bold")

# Count plot
sns.countplot(x="Exited", data=df, palette=["#4CAF50", "#F44336"], ax=axes[1])
axes[1].set_title("Churn Count", fontsize=13, fontweight="bold")
axes[1].set_xticklabels(["Retained", "Churned"])
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig("plots/01_churn_distribution.png", bbox_inches="tight")
plt.show()

# --- 3.3 Numerical Features Distribution ---
num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "EstimatedSalary"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(
        data=df, x=col, hue="Exited",
        palette=["#4CAF50", "#F44336"],
        bins=30, ax=axes[i], alpha=0.7
    )
    axes[i].set_title(f"{col} by Churn", fontweight="bold")
    axes[i].set_xlabel("")

plt.suptitle("Numerical Features Distribution by Churn", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/02_numerical_distribution.png", bbox_inches="tight")
plt.show()

# --- 3.4 Categorical Features vs Churn ---
cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)["Exited"].mean().reset_index()
    churn_rate.columns = [col, "Churn Rate"]
    sns.barplot(x=col, y="Churn Rate", data=churn_rate,
                palette="RdYlGn_r", ax=axes[i])
    axes[i].set_title(f"Churn Rate by {col}", fontweight="bold")
    axes[i].set_ylim(0, 0.6)
    for p in axes[i].patches:
        axes[i].annotate(f"{p.get_height():.1%}",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=10)

plt.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_categorical_churn_rate.png", bbox_inches="tight")
plt.show()

# --- 3.5 Correlation Heatmap ---
plt.figure(figsize=(10, 6))
corr = df[num_cols + ["Exited"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# ==============================================================================
# 4. FEATURE ENGINEERING & PREPROCESSING
# ==============================================================================
print("\n" + "="*50)
print("4. FEATURE ENGINEERING & PREPROCESSING")
print("="*50)

df_model = df.copy()

# Drop kolom yang tidak relevan
df_model.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Encode categorical
le = LabelEncoder()
df_model["Gender"]    = le.fit_transform(df_model["Gender"])     # Female=0, Male=1
df_model["Geography"] = le.fit_transform(df_model["Geography"])  # France=0, Germany=1, Spain=2

# Feature Engineering — tambah fitur baru
df_model["BalanceSalaryRatio"]   = df_model["Balance"] / (df_model["EstimatedSalary"] + 1)
df_model["AgeGroup"]             = pd.cut(df_model["Age"],
                                          bins=[0, 30, 45, 60, 100],
                                          labels=[0, 1, 2, 3]).astype(int)
df_model["IsZeroBalance"]        = (df_model["Balance"] == 0).astype(int)
df_model["ProductsPerTenure"]    = df_model["NumOfProducts"] / (df_model["Tenure"] + 1)

print(f"Features after engineering: {df_model.shape[1] - 1} features")
print(df_model.head(3))

# ==============================================================================
# 5. TRAIN-TEST SPLIT & SMOTE
# ==============================================================================
print("\n" + "="*50)
print("5. TRAIN-TEST SPLIT & SMOTE")
print("="*50)

FEATURES = [c for c in df_model.columns if c != "Exited"]
TARGET   = "Exited"

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn rate train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

# SMOTE — handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Train size: {X_train_sm.shape[0]} samples")
print(f"  Churn rate: {y_train_sm.mean():.1%} (balanced!)")

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# ==============================================================================
# 6. MODEL TRAINING & COMPARISON
# ==============================================================================
print("\n" + "="*50)
print("6. MODEL TRAINING & COMPARISON")
print("="*50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=5, random_state=42,
                                         eval_metric="logloss", verbosity=0),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Pakai scaled data untuk Logistic Regression, unscaled untuk tree-based
    X_tr = X_train_sc if name == "Logistic Regression" else X_train_sm
    X_te = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train_sm)
    y_pred     = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]

    cv_scores = cross_val_score(model, X_tr, y_train_sm,
                                cv=cv, scoring="roc_auc", n_jobs=-1)

    results[name] = {
        "model":       model,
        "y_pred":      y_pred,
        "y_proba":     y_pred_proba,
        "roc_auc":     roc_auc_score(y_test, y_pred_proba),
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "X_test_used": X_te,
    }

    print(f"\n{name}")
    print(f"  ROC-AUC (test): {results[name]['roc_auc']:.4f}")
    print(f"  ROC-AUC (CV):   {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Retained", "Churned"]))

# ==============================================================================
# 7. MODEL EVALUATION — VISUALIZATIONS
# ==============================================================================
print("\n" + "="*50)
print("7. MODEL EVALUATION VISUALIZATION")
print("="*50)

# --- 7.1 ROC Curve ---
plt.figure(figsize=(8, 6))
colors = ["#2196F3", "#4CAF50", "#F44336"]

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)

plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Comparison", fontsize=13, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/05_roc_curve.png", bbox_inches="tight")
plt.show()

# --- 7.2 Confusion Matrix (best model = XGBoost) ---
best_name = max(results, key=lambda x: results[x]["roc_auc"])
best_res  = results[best_name]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix - counts
cm = confusion_matrix(y_test, best_res["y_pred"])
ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues"
)
axes[0].set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")

# Confusion matrix - normalized
cm_norm = confusion_matrix(y_test, best_res["y_pred"], normalize="true")
ConfusionMatrixDisplay(cm_norm, display_labels=["Retained", "Churned"]).plot(
    ax=axes[1], colorbar=False, cmap="Blues", values_format=".1%"
)
axes[1].set_title(f"Normalized Confusion Matrix\n{best_name}", fontweight="bold")

plt.tight_layout()
plt.savefig("plots/06_confusion_matrix.png", bbox_inches="tight")
plt.show()

# --- 7.3 Precision-Recall Curve ---
plt.figure(figsize=(8, 5))
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    plt.plot(rec, prec, label=name, color=color, lw=2)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("plots/07_precision_recall.png", bbox_inches="tight")
plt.show()

# --- 7.4 Model Comparison Bar Chart ---
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "ROC-AUC": [r["roc_auc"] for r in results.values()],
    "CV Mean": [r["cv_mean"] for r in results.values()],
}).sort_values("ROC-AUC", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(comparison_df["Model"], comparison_df["ROC-AUC"],
               color=["#4CAF50" if v == comparison_df["ROC-AUC"].max() else "#90CAF9"
                      for v in comparison_df["ROC-AUC"]])

for bar, val in zip(bars, comparison_df["ROC-AUC"]):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="right", color="white", fontweight="bold")

ax.set_xlim(0.5, 1.0)
ax.set_title("Model Comparison — ROC-AUC Score", fontsize=13, fontweight="bold")
ax.set_xlabel("ROC-AUC Score")
plt.tight_layout()
plt.savefig("plots/08_model_comparison.png", bbox_inches="tight")
plt.show()

print(f"\n🏆 Best Model: {best_name} (ROC-AUC: {best_res['roc_auc']:.4f})")

# ==============================================================================
# 8. SHAP VALUES — MODEL INTERPRETABILITY
# ==============================================================================
print("\n" + "="*50)
print("8. SHAP VALUES — MODEL INTERPRETABILITY")
print("="*50)

best_model = results[best_name]["model"]
X_test_used = results[best_name]["X_test_used"]

# SHAP explainer
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_used)

# --- 8.1 SHAP Summary Plot (Bar) ---
plt.figure()
shap.summary_plot(shap_values, X_test_used,
                  feature_names=FEATURES,
                  plot_type="bar", show=False)
plt.title(f"SHAP Feature Importance — {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/09_shap_bar.png", bbox_inches="tight")
plt.show()

# --- 8.2 SHAP Summary Plot (Beeswarm) ---
plt.figure()
shap.summary_plot(shap_values, X_test_used,
                  feature_names=FEATURES, show=False)
plt.title(f"SHAP Beeswarm — {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/10_shap_beeswarm.png", bbox_inches="tight")
plt.show()

# --- 8.3 SHAP Waterfall (satu sample) ---
print("\nContoh prediksi churn untuk 1 customer:")
sample_idx = 0
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test_used[sample_idx] if hasattr(X_test_used, 'iloc')
             else X_test_used[sample_idx],
        feature_names=FEATURES
    )
)

# ==============================================================================
# 9. BUSINESS INSIGHTS & RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*50)
print("9. BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*50)

# Feature importance dari SHAP
shap_importance = pd.DataFrame({
    "Feature":   FEATURES,
    "SHAP Mean": np.abs(shap_values).mean(axis=0)
}).sort_values("SHAP Mean", ascending=False)

print("\nTop 5 Most Important Features:")
print(shap_importance.head(5).to_string(index=False))

print("""
📊 Key Business Insights:
─────────────────────────────────────────────────────
1. AGE is a strong churn predictor — older customers (45-60)
   are significantly more likely to churn.
   → Recommendation: Create loyalty programs targeting older segments.

2. ACTIVE MEMBERSHIP drastically reduces churn.
   → Recommendation: Encourage inactive members to engage with
     exclusive offers or personalized outreach.

3. GERMANY customers show higher churn rates than France/Spain.
   → Recommendation: Investigate service quality issues in Germany
     and tailor retention strategies regionally.

4. BALANCE=0 customers have higher churn risk.
   → Recommendation: Identify zero-balance customers early and
     offer incentives to increase engagement.

5. NUMBER OF PRODUCTS has a non-linear effect:
   customers with 3-4 products churn more than those with 1-2.
   → Recommendation: Avoid aggressive cross-selling to at-risk customers.
─────────────────────────────────────────────────────
""")

# ==============================================================================
# 10. SAVE MODEL
# ==============================================================================
import pickle, os

os.makedirs("model", exist_ok=True)

with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print(f"✅ Model saved to model/best_model.pkl")
print(f"✅ Best model: {best_name} | ROC-AUC: {best_res['roc_auc']:.4f}")
print("\n🎉 Analysis complete! Check the 'plots/' folder for all visualizations.")
