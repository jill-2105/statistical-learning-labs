<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Statistical Learning Assignment 3 - ISLR Python Implementation

This repository provides complete Python solutions to three key exercises from *An Introduction to Statistical Learning* (ISLR). Each question follows the exact problem specifications, including data preparation, exploratory analysis, model fitting, validation strategies, and performance evaluation on the Auto, Default, and College datasets.[^1][^2][^3][^4]

## 📋 Problem Descriptions

### Question 1 (14 parts) - MPG Classification

**Objective**: Develop classifiers to predict high/low gas mileage (`mpg01`: 1=above median MPG, 0=below) using Auto dataset.
**Tasks Solved**:

- (a) Created binary `mpg01` variable
- (b) EDA: Boxplots/scatterplots identified `horsepower`, `weight`, `displacement`, `acceleration` as key predictors
- (c) 70/30 train/test split (`random_state=42`)
- (d)-(g) LDA, QDA, Logistic Regression, Naive Bayes → test errors reported
- (h) KNN (K=1-20) → optimal K=3 at **86.4% accuracy**[^2]


### Question 2 (5 parts) - Default Prediction Validation

**Objective**: Estimate logistic regression test error for credit default using validation set approach.
**Tasks Solved**:

- (a) Base model: `income` + `balance` → 97.33% accuracy (seed=42)
- (b) Validation set error estimation (70/30 split)
- (c) Repeated 3x (seeds: 42, 123, 456) → **mean 97.38% accuracy**, SD=0.04%
- (d) Extended model (+`student` dummy) → slightly worse (2.70% error vs 2.62%)[^3]


### Question 3 (9 parts) - College Applications Regression

**Objective**: Compare regression methods to predict applications (`Apps`) using College dataset predictors.
**Tasks Solved**:

- (a) 70/30 train/test split (`random_state=42`)
- (b) OLS baseline: RMSE=1389.89
- (c) Ridge (CV λ=0.0010): RMSE=1389.88
- (d) Lasso (CV λ=3.78, **17 non-zero coeffs**): RMSE=1388.44
- (e) PCR (CV M=17): RMSE=1389.89
- (f) **PLS (CV M=13, best RMSE=1383.69)**
- (g) PLS most accurate; Lasso provides sparsity[^4]


## 📁 Repository Structure

```
Assignment3/
├── Q1/                          # Auto MPG Classification (Question 1)
│   ├── data/Auto.csv
│   ├── plots/                   # EDA + KNN accuracy curve
│   ├── question1_analysis.py
│   └── question1_report.md
├── Q2/                          # Default Logistic Validation (Question 2)
│   ├── data/Default.csv
│   ├── plots/                   # Validation error bars
│   ├── question2_analysis.py
│   └── question2_report.md
├── Q3/                          # College Regression Comparison (Question 3)
│   ├── data/College.csv
│   ├── plots/                   # RMSE bars + CV curves
│   ├── question3_analysis.py
│   └── question3_report.md
├── Team members.txt
└── README.md
```


## 📊 Performance Summary

| Question | Best Model | Test Metric | Key Insight |
| :-- | :-- | :-- | :-- |
| Q1 (Classification) | **KNN K=3** | **86.4% Accuracy** | Beats parametric models; horsepower/weight dominant[^2] |
| Q2 (Logistic) | income+balance | **2.62% Error** | Student dummy hurts performance; stable across splits[^3] |
| Q3 (Regression) | **PLS M=13** | **1383.69 RMSE** | Best prediction; Lasso (17 vars) interpretable alternative[^4] |

## 👥 Team

- **Jenish Modi**
- **Bardiya Rasekh**
- **Jaber Al Siam**
- **Jill Patel**[^1]


## 🛠️ Setup \& Execution

```bash
# Python 3.10+ required
pip install numpy pandas matplotlib seaborn scikit-learn

# Run from project root
python Q1/question1_analysis.py    # Generates Q1 plots + metrics
python Q2/question2_analysis.py    # Q2 validation results
python Q3/question3_analysis.py    # Q3 regression comparison
```

**Reproducibility**: Fixed seeds (42), local data files, 10-fold CV where specified.[^1]

## 🔍 Key Visualizations Generated

- **Q1**: Horsepower/weight boxplots, feature scatter, KNN accuracy vs K
- **Q2**: Validation error comparison bars (3 seeds), confusion matrices
- **Q3**: Test RMSE bar chart, PCR/PLS CV curves, Lasso coefficients[^2][^3][^4]


## 📄 License

MIT License - Educational use encouraged. PRs welcome for enhancements (class weights, grid search).[^1]

***

*Completed: December 2025 | ISLR Assignment 3 | 100% problem coverage*

<div align="center">⁂</div>

[^1]: README.md

[^2]: question1_report.md

[^3]: question2_report.md

[^4]: question3_report.mdgti add README>md