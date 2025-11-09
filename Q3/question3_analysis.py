#===================================================================================================================================
# Assignment 3 - Question 3
# Question-6.9 Page No. 294
# In this problem, we will predict the number of applications received using the other variables in the College data set.
# Models: OLS, Ridge, Lasso, PCR, PLS
#===================================================================================================================================

# --- Imports ---
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# --- Ensure plots folder exists ---
os.makedirs("plots", exist_ok=True)

# --- Step (a) Load College dataset (local, reproducible) ---
def load_college_dataset():
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)
    local_csv = data_dir / "College.csv"

    if not local_csv.exists():
        raise FileNotFoundError(
            "data/College.csv not found.\n"
            "Place the ISLR College.csv at: " + local_csv.as_posix()
        )

    df = pd.read_csv(local_csv)
    # Some versions have first column as college name (Unnamed: 0)
    if df.columns[0].lower() in {"unnamed: 0", "name"}:
        df = df.rename(columns={df.columns[0]: "College"})
    return df

College = load_college_dataset()

print("\nDataset Overview:")
print(College.head())
print(f"\nDataset shape: {College.shape}")

# --- Step (b) Prepare features/target ---
target_col = "Apps" if "Apps" in College.columns else "apps"
assert target_col in College.columns, f"Apps column not found. Columns: {College.columns.tolist()}"

df = College.copy()

# Encode Private as 1/0 if present
if "Private" in df.columns:
    df["Private"] = (df["Private"].astype(str).str.strip().str.lower() == "yes").astype(int)

# Drop non-numeric identifier if present
if "College" in df.columns:
    df = df.drop(columns=["College"])

y = df[target_col].values
X = df.drop(columns=[target_col]).values
feature_names = [c for c in df.columns if c != target_col]

# Train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Standardize predictors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# --- Helper to collect results ---
results = {}

def add_result(model_name, y_true, y_pred, extra=None):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    results[model_name] = {"MSE": float(mse), "RMSE": rmse}
    if extra is not None:
        results[model_name].update(extra)

# --- Step (c) OLS ---
print("\n--- OLS ---")
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)
add_result("OLS", y_test, y_pred_ols)
print(f"OLS Test RMSE: {results['OLS']['RMSE']:.2f}")

# --- Step (d) Ridge Regression (CV) ---
print("\n--- Ridge Regression (CV) ---")
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=10, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)
add_result("Ridge", y_test, y_pred_ridge, extra={"alpha": float(ridge_cv.alpha_)})
print(f"Ridge best alpha: {ridge_cv.alpha_:.4f}")
print(f"Ridge Test RMSE: {results['Ridge']['RMSE']:.2f}")

# --- Step (e) Lasso (CV) ---
print("\n--- Lasso (CV) ---")
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)
nonzero = int(np.sum(lasso_cv.coef_ != 0))
nonzero_features = [feature_names[i] for i, coef in enumerate(lasso_cv.coef_) if coef != 0]
add_result(
    "Lasso", y_test, y_pred_lasso,
    extra={"alpha": float(lasso_cv.alpha_), "nonzero": nonzero, "features": nonzero_features}
)
print(f"Lasso best alpha: {lasso_cv.alpha_:.6f}")
print(f"Lasso non-zero coefficients: {nonzero}")
print(f"Lasso Test RMSE: {results['Lasso']['RMSE']:.2f}")

# --- Step (f) PCR (Principal Component Regression) ---
print("\n--- PCR (CV over components) ---")
p = X_train.shape[1]
kf = KFold(n_splits=10, shuffle=True, random_state=42)

comp_candidates = np.arange(1, min(p, 20) + 1)  # cap components at 20
pcr_cv_errors = []
for m in comp_candidates:
    fold_mse = []
    for tr_idx, va_idx in kf.split(X_train):
        pca = PCA(n_components=m, random_state=42)
        Ztr = pca.fit_transform(X_train[tr_idx])
        Zva = pca.transform(X_train[va_idx])
        lr = LinearRegression().fit(Ztr, y_train[tr_idx])
        pred = lr.predict(Zva)
        fold_mse.append(mean_squared_error(y_train[va_idx], pred))
    pcr_cv_errors.append(np.mean(fold_mse))

best_m_pcr = int(comp_candidates[np.argmin(pcr_cv_errors)])
# Fit final PCR with best M
pca_best = PCA(n_components=best_m_pcr, random_state=42)
Ztr_best = pca_best.fit_transform(X_train)
Zte_best = pca_best.transform(X_test)
lr_best = LinearRegression().fit(Ztr_best, y_train)
y_pred_pcr = lr_best.predict(Zte_best)
add_result("PCR", y_test, y_pred_pcr, extra={"components": best_m_pcr})
print(f"PCR chosen components (M): {best_m_pcr}")
print(f"PCR Test RMSE: {results['PCR']['RMSE']:.2f}")

# --- Step (g) PLS (Partial Least Squares) ---
print("\n--- PLS (CV over components) ---")
pls_cv_errors = []
for m in comp_candidates:
    fold_mse = []
    for tr_idx, va_idx in kf.split(X_train):
        pls = PLSRegression(n_components=m)
        pls.fit(X_train[tr_idx], y_train[tr_idx])
        pred = pls.predict(X_train[va_idx]).ravel()
        fold_mse.append(mean_squared_error(y_train[va_idx], pred))
    pls_cv_errors.append(np.mean(fold_mse))

best_m_pls = int(comp_candidates[np.argmin(pls_cv_errors)])
pls_best = PLSRegression(n_components=best_m_pls)
pls_best.fit(X_train, y_train)
y_pred_pls = pls_best.predict(X_test).ravel()
add_result("PLS", y_test, y_pred_pls, extra={"components": best_m_pls})
print(f"PLS chosen components (M): {best_m_pls}")
print(f"PLS Test RMSE: {results['PLS']['RMSE']:.2f}")

# --- Step (h) Plot comparisons ---

# 1) RMSE comparison
labels = ["OLS", "Ridge", "Lasso", "PCR", "PLS"]
rmses = [results[m]["RMSE"] for m in labels]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, rmses, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"], alpha=0.9)
plt.ylabel("Test RMSE")
plt.title("Test RMSE Comparison: OLS, Ridge, Lasso, PCR, PLS")
plt.grid(axis='y', alpha=0.3)
for b, v in zip(bars, rmses):
    plt.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
plt.savefig("plots/q3_test_rmse_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 2) CV curves for PCR and PLS
plt.figure(figsize=(10,6))
plt.plot(comp_candidates, np.sqrt(pcr_cv_errors), marker='o', label="PCR CV RMSE")
plt.plot(comp_candidates, np.sqrt(pls_cv_errors), marker='s', label="PLS CV RMSE")
plt.axvline(best_m_pcr, color="#4C78A8", linestyle="--", alpha=0.6, label=f"PCR M={best_m_pcr}")
plt.axvline(best_m_pls, color="#E45756", linestyle="--", alpha=0.6, label=f"PLS M={best_m_pls}")
plt.xlabel("Number of Components (M)")
plt.ylabel("CV RMSE")
plt.title("PCR and PLS CV Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("plots/q3_cv_curves_pcr_pls.png", dpi=300, bbox_inches='tight')
plt.show()

# --- Step (i) Print summary ---
print("\n" + "="*50)
print("SUMMARY OF TEST RESULTS (RMSE)")
print("="*50)
for name in labels:
    line = f"{name}: RMSE={results[name]['RMSE']:.2f}"
    if name == "Ridge":
        line += f", alpha={results[name]['alpha']:.4f}"
    if name == "Lasso":
        line += f", alpha={results[name]['alpha']:.6f}, nonzero={results[name]['nonzero']}"
    if name in {"PCR", "PLS"}:
        line += f", components={results[name]['components']}"
    print(line)

print("\nLasso selected features (non-zero):")
print(results["Lasso"]["features"])

print("\nPlots saved successfully:")
print(" - plots/q3_test_rmse_comparison.png")
print(" - plots/q3_cv_curves_pcr_pls.png")
