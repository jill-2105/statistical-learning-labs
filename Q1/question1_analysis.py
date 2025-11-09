#=================================================================================================================================
# Assignment 3 - Question 1
# Question-4.14 Page No. 206
# In this problem, you will develop a model to predict whether a given car gets high or low gas mileage based on the Auto data set.
#=================================================================================================================================

# --- Imports ---
import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Ensure plots folder exists ---
os.makedirs("plots", exist_ok=True)

# --- Step (a) Load Auto dataset (local) ---
def load_auto_dataset():
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)
    local_csv = data_dir / "Auto.csv"
    if not local_csv.exists():
        raise FileNotFoundError(
            "data/Auto.csv not found. Place the ISLR Auto.csv at: " + local_csv.as_posix()
        )
    # Handle '?' and missing values
    df = pd.read_csv(local_csv, na_values='?').dropna()
    return df

Auto = load_auto_dataset()

print("Dataset Overview:")
print(Auto.head())
print(f"\nDataset shape: {Auto.shape}")

# --- Step (b) Create binary target mpg01 ---
Auto["mpg01"] = (Auto["mpg"] > Auto["mpg"].median()).astype(int)

# --- Step (c) Exploratory plots ---
plt.figure(figsize=(8,6))
sns.boxplot(x="mpg01", y="horsepower", data=Auto)
plt.title("Horsepower vs MPG01")
plt.savefig("plots/boxplot_horsepower.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(x="mpg01", y="weight", data=Auto)
plt.title("Weight vs MPG01")
plt.savefig("plots/boxplot_weight.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(x="horsepower", y="weight", hue="mpg01", data=Auto, palette="coolwarm")
plt.title("Horsepower vs Weight colored by MPG01")
plt.savefig("plots/scatter_horsepower_weight.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Step (d) Train/Test split ---
features = ["horsepower", "weight", "displacement", "acceleration"]
X = Auto[features]
y = Auto["mpg01"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# --- Step (e) LDA ---
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
lda_acc = accuracy_score(y_test, lda_pred)
print(f"LDA Test Accuracy: {lda_acc:.3f}")

# --- Step (f) QDA ---
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
qda_acc = accuracy_score(y_test, qda_pred)
print(f"QDA Test Accuracy: {qda_acc:.3f}")

# --- Step (g) Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print(f"Logistic Regression Test Accuracy: {log_acc:.3f}")

# --- Step (h) Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Test Accuracy: {nb_acc:.3f}")

# --- Step (i) KNN over K=1..20 ---
knn_results = {}
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    knn_results[k] = acc

# Plot KNN accuracy vs K
plt.figure(figsize=(8,6))
plt.plot(list(knn_results.keys()), list(knn_results.values()), marker='o')
plt.xlabel("K value")
plt.ylabel("Test Accuracy")
plt.title("KNN Test Accuracy vs K")
plt.grid(True)
plt.savefig("plots/knn_accuracy_vs_k.png", dpi=300, bbox_inches='tight')
plt.show()

best_k = max(knn_results, key=knn_results.get)
print(f"Best K value: {best_k} with Accuracy: {knn_results[best_k]:.3f}")

# --- Step (j) Optional: confusion matrix for best model (KNN) ---
best_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
best_pred = best_knn.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix (KNN best K):")
print(cm)

# --- Summary ---
print("\nSummary of Test Accuracies:")
print(f"LDA: {lda_acc:.3f}")
print(f"QDA: {qda_acc:.3f}")
print(f"Logistic Regression: {log_acc:.3f}")
print(f"Naive Bayes: {nb_acc:.3f}")
print(f"KNN (best K={best_k}): {knn_results[best_k]:.3f}")

print("\nPlots saved successfully:")
print(" - plots/boxplot_horsepower.png")
print(" - plots/boxplot_weight.png")
print(" - plots/knn_accuracy_vs_k.png")
print(" - plots/scatter_horsepower_weight.png")
