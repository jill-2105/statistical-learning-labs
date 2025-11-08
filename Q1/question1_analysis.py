# =======================
# Assignment 3 - Question 1
# Predicting whether a car has high or low MPG
# =======================

# --- Imports ---
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- Ensure plots folder exists ---
os.makedirs("plots", exist_ok=True)

# --- Step (a) Load data and create mpg01 variable ---
# Dataset available at https://www.statlearning.com/resources-first-edition
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
Auto = pd.read_csv(url, na_values='?').dropna()

# Binary variable: 1 if mpg > median, else 0
mpg01 = (Auto['mpg'] > Auto['mpg'].median()).astype(int)
Auto['mpg01'] = mpg01

print(Auto.head())

# --- Step (b) Explore data graphically ---
plt.figure(figsize=(8,6))
sns.boxplot(x='mpg01', y='horsepower', data=Auto)
plt.title("Horsepower vs MPG01")
plt.savefig("plots/boxplot_horsepower.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='mpg01', y='weight', data=Auto)
plt.title("Weight vs MPG01")
plt.savefig("plots/boxplot_weight.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='horsepower', y='weight', hue='mpg01', data=Auto, palette='coolwarm')
plt.title("Horsepower vs Weight colored by MPG01")
plt.savefig("plots/scatter_horsepower_weight.png", dpi=300, bbox_inches='tight')
plt.show()

# Based on visual inspection, horsepower, weight, and displacement are strong predictors

# --- Step (c) Split into training and test sets ---
features = ['horsepower', 'weight', 'displacement', 'acceleration']
X = Auto[features]
y = Auto['mpg01']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step (d) LDA ---
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
lda_acc = accuracy_score(y_test, lda_pred)
print(f"LDA Test Accuracy: {lda_acc:.3f}")

# --- Step (e) QDA ---
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_pred = qda.predict(X_test)
qda_acc = accuracy_score(y_test, qda_pred)
print(f"QDA Test Accuracy: {qda_acc:.3f}")

# --- Step (f) Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print(f"Logistic Regression Test Accuracy: {log_acc:.3f}")

# --- Step (g) Naive Bayes ---
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Test Accuracy: {nb_acc:.3f}")

# --- Step (h) KNN for multiple K values ---
knn_results = {}
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = accuracy_score(y_test, pred)
    knn_results[k] = acc

# Plot KNN results
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

# --- Summary of results ---
print("\nSummary of Test Accuracies:")
print(f"LDA: {lda_acc:.3f}")
print(f"QDA: {qda_acc:.3f}")
print(f"Logistic Regression: {log_acc:.3f}")
print(f"Naive Bayes: {nb_acc:.3f}")
print(f"KNN (best K={best_k}): {knn_results[best_k]:.3f}")
