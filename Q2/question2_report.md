# Question 2: Logistic Regression with Validation (Default Dataset)

## Objective
Estimate the test error of a logistic regression model predicting credit default using `income` and `balance`, then assess whether adding a `student` indicator improves performance.

## Data Preparation
- Dataset: ISLR “Default” (local copy placed in `Q2/data/Default.csv` for reproducibility).
- Variables:
  - `default` (Yes/No) → `default_binary` (1 for Yes, 0 for No)
  - `student` (Yes/No) → `student_binary` (1 for Yes, 0 for No)
  - `balance` (numeric), `income` (numeric)
- Class balance: Yes = 3.33%, No = 96.67% (highly imbalanced).

## Methodology
- Validation set approach with 70/30 train/validation splits.
- Base model: Logistic Regression with predictors `income` and `balance`.
- Repeated the validation split for 3 different seeds: 42, 123, 456.
- Extended model: Added `student_binary` as an additional predictor.
- Reported accuracy, error, range, and standard deviation across splits.
- Saved comparison plot to `plots/validation_error_comparison.png`.

## Results

### Single Split (seed=42)
- Accuracy: 97.33%
- Error: 2.67%

Confusion Matrix (income + balance):

|        | Pred No | Pred Yes |
|---|---:|---:|
| True No | 2896 | 10 |
| True Yes | 70 | 24 |

### Multiple Splits (income + balance)

| Metric | Value |
|---|---|
| Split 42 Accuracy | 97.33% |
| Split 123 Accuracy | 97.43% |
| Split 456 Accuracy | 97.37% |
| Mean Accuracy | 97.38% |
| Mean Error | 2.62% |
| Error Std Dev | 0.04% |
| Error Range | [2.57%, 2.67%] |

### With Student Variable (income + balance + student)

| Metric | Value |
|---|---|
| Split 42 Accuracy | 97.33% |
| Split 123 Accuracy | 97.27% |
| Split 456 Accuracy | 97.30% |
| Mean Accuracy | 97.30% |
| Mean Error | 2.70% |
| Error Std Dev | 0.03% |
| Error Range | [2.67%, 2.73%] |

### Plot
- See `plots/validation_error_comparison.png` for side-by-side error bars across seeds.

## Conclusions
- Best specification: `income + balance` with mean validation error of 2.62%.
- Adding `student` slightly worsened performance (mean error increased by 0.08 percentage points), indicating it does not add predictive value in this setup.
- Confusion matrix shows strong performance but with some false negatives due to class imbalance; thresholds or class-weighting could be explored in future work.
- The results are stable across random splits, suggesting low variance in estimates.

## Reproducibility Notes
- Data loaded locally from `data/Default.csv` to avoid upstream access issues while using the official ISLR dataset.
- Random seeds used: 42, 123, 456.
- Environment: numpy, pandas, scikit-learn, matplotlib.