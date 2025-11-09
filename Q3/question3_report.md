# Question 3: College Dataset — Predicting Applications

## Objective
Compare multiple regression approaches to predict the number of applications received by colleges using the ISLR College dataset. Models evaluated: OLS, Ridge, Lasso, PCR, and PLS.

## Data Preparation
- Dataset: ISLR “College” (local copy at `Q3/data/College.csv`).
- Target: `Apps` (applications).
- Predictors: All remaining variables; `Private` encoded as 1 (Yes) / 0 (No).
- Train/Test split: 70/30 with `random_state=42`.
- Predictors standardized for all models.

## Methods
- OLS via LinearRegression.
- Ridge with 10-fold CV over 100 alphas on log scale.
- Lasso with 10-fold CV; reported best alpha and non-zero coefficients.
- PCR: 10-fold CV over components M=1..20; refit with best M.
- PLS: 10-fold CV over components M=1..20; refit with best M.
- Metrics: Test RMSE and MSE; plots saved for model comparison and CV curves.

## Results

### Test RMSE Summary

| Model | Test RMSE | Notes |
|---|---:|---|
| OLS | 1389.89 | Baseline linear fit |
| Ridge | 1389.88 | Best alpha = 0.0010 |
| Lasso | 1388.44 | Best alpha = 3.784174; Non-zero = 17 |
| PCR | 1389.89 | Components M = 17 |
| PLS | 1383.69 | Components M = 13 |

- RMSE comparison plot: `plots/q3_test_rmse_comparison.png`
- CV curves for PCR/PLS: `plots/q3_cv_curves_pcr_pls.png`

### Lasso Selected Predictors
The following features retained non-zero coefficients under the best alpha:
`Private`, `Accept`, `Enroll`, `Top10perc`, `Top25perc`, `F.Undergrad`, `P.Undergrad`, `Outstate`, `Room.Board`, `Books`, `Personal`, `PhD`, `Terminal`, `S.F.Ratio`, `perc.alumni`, `Expend`, `Grad.Rate`.

## Discussion
- Best performing model: PLS with RMSE = 1383.69, outperforming OLS/Ridge/PCR and slightly better than Lasso. The benefit suggests that learning latent components that jointly maximize covariance with the response is effective for this dataset.
- Lasso achieved RMSE = 1388.44 and produced a sparse, interpretable model with 17 predictors, highlighting variables such as `Outstate`, `Room.Board`, `perc.alumni`, and `Expend`.
- Ridge and PCR performed close to OLS, indicating limited gains from pure shrinkage or unsupervised dimensionality reduction here.
- Component choices selected by CV were M=17 (PCR) and M=13 (PLS), balancing bias-variance while capturing structure in the predictors.

## Reproducibility
- Data loaded from `data/College.csv`.
- Random seed: 42 for train/test split; 10-fold CV with shuffling.
- Libraries: numpy, pandas, scikit-learn, matplotlib.
