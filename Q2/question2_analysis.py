# =======================
# Assignment 3 - Question 2
# Logistic Regression with Validation Set Approach
# Default Dataset (ISLR)
# =======================git st


# --- Imports ---
import os
import pathlib
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# --- Ensure plots folder exists ---
os.makedirs("plots", exist_ok=True)


# --- Step (a) Load Default dataset with official-URL-first + local fallback ---
def load_default_dataset():
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)
    local_csv = data_dir / "Default.csv"

    official = "https://www.statlearning.com/s/Default.csv"
    mirror   = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Default.csv"

    if not local_csv.exists():
        try:
            print("Attempting download from official ISLR URL ...")
            urllib.request.urlretrieve(official, local_csv.as_posix())
            print("Downloaded from official ISLR URL.")
        except Exception as e_off:
            print(f"Official URL failed ({e_off}). Trying mirror ...")
            try:
                urllib.request.urlretrieve(mirror, local_csv.as_posix())
                print("Downloaded from public mirror (identical schema).")
            except Exception as e_mir:
                raise RuntimeError(
                    "Failed to retrieve Default.csv from both sources.\n"
                    f"Official error: {e_off}\nMirror error: {e_mir}"
                )
    else:
        print("Using local cached data/Default.csv")

    df = pd.read_csv(local_csv)
    # Sanity checks and binary encodings
    required_cols = {"default", "student", "balance", "income"}
    assert required_cols.issubset(df.columns), f"Unexpected columns. Found: {df.columns.tolist()}"
    df["default_binary"] = (df["default"] == "Yes").astype(int)
    df["student_binary"] = (df["student"] == "Yes").astype(int)
    return df


Default = load_default_dataset()

print("\nDataset Overview:")
print(Default.head())
print(f"\nDataset shape: {Default.shape}")
print(f"\nColumn names: {Default.columns.tolist()}")
print("\nDefault rate (Yes/No proportion):")
print(Default["default"].value_counts(normalize=True))


# --- Step (b) Logistic regression: income + balance (single split) ---
print("\n--- Single Split Validation (income + balance) ---")
X = Default[["income", "balance"]]
y = Default["default_binary"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_val)

val_acc = accuracy_score(y_val, y_pred)
val_error = 1 - val_acc
cm = confusion_matrix(y_val, y_pred)

print(f"Validation Set Accuracy: {val_acc:.4f}")
print(f"Validation Set Error:    {val_error:.4f}")
print("Confusion Matrix (income + balance):")
print(cm)


# --- Step (c) Repeat with 3 different random splits ---
print("\n--- Multiple Random Splits (income + balance) ---")
random_seeds = [42, 123, 456]
errors = []
accuracies = []

for seed in random_seeds:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=seed)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    err = 1 - acc
    accuracies.append(acc)
    errors.append(err)
    print(f"Split {seed} - Accuracy: {acc:.4f} | Error: {err:.4f}")

print(f"\nMean Validation Accuracy: {np.mean(accuracies):.4f}")
print(f"Mean Validation Error:    {np.mean(errors):.4f}")
print(f"Std Deviation (Error):    {np.std(errors):.4f}")
print(f"Error Range:              [{np.min(errors):.4f}, {np.max(errors):.4f}]")


# --- Step (d) Add student dummy variable ---
print("\n--- Logistic Regression with Student Variable ---")
X_with_student = Default[["income", "balance", "student_binary"]]
errors_with_student = []
accuracies_with_student = []

for seed in random_seeds:
    X_train, X_val, y_train, y_val = train_test_split(X_with_student, y, test_size=0.30, random_state=seed)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    err = 1 - acc
    accuracies_with_student.append(acc)
    errors_with_student.append(err)
    print(f"Split {seed} (with student) - Accuracy: {acc:.4f} | Error: {err:.4f}")

print(f"\nMean Validation Accuracy (with student): {np.mean(accuracies_with_student):.4f}")
print(f"Mean Validation Error (with student):    {np.mean(errors_with_student):.4f}")
print(f"Std Deviation (Error, with student):     {np.std(errors_with_student):.4f}")
print(f"Error Range (with student):              [{np.min(errors_with_student):.4f}, {np.max(errors_with_student):.4f}]")


# --- Summary Comparison ---
print("\n" + "="*50)
print("SUMMARY OF RESULTS")
print("="*50)
print("\nModel: income + balance")
print(f"  Mean Validation Accuracy: {np.mean(accuracies):.4f}")
print(f"  Mean Validation Error:    {np.mean(errors):.4f}")
print(f"  Error Range:              [{np.min(errors):.4f}, {np.max(errors):.4f}]")

print("\nModel: income + balance + student")
print(f"  Mean Validation Accuracy: {np.mean(accuracies_with_student):.4f}")
print(f"  Mean Validation Error:    {np.mean(errors_with_student):.4f}")
print(f"  Error Range:              [{np.min(errors_with_student):.4f}, {np.max(errors_with_student):.4f}]")

improvement = np.mean(errors) - np.mean(errors_with_student)
print(f"\nImprovement from adding 'student': {improvement:.4f}")
if improvement > 0:
    print("Adding 'student' variable REDUCES validation error.")
elif improvement < 0:
    print("Adding 'student' variable INCREASES validation error.")
else:
    print("Adding 'student' variable yields NO change in validation error.")


# --- Optional: Plot error comparison ---
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(random_seeds))
width = 0.35

plt.bar(x_pos - width/2, errors, width, label='income + balance', alpha=0.85)
plt.bar(x_pos + width/2, errors_with_student, width, label='income + balance + student', alpha=0.85)

plt.xlabel('Random Seed')
plt.ylabel('Validation Error')
plt.title('Validation Error Comparison Across Different Splits')
plt.xticks(x_pos, random_seeds)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig("plots/validation_error_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Plot saved to plots/validation_error_comparison.png")
