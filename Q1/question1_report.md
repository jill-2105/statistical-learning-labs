# Question 1: Auto Dataset Classification Analysis

## Objective
Predict whether a car gets high or low miles per gallon (MPG) using various classification methods including LDA, QDA, Logistic Regression, Naive Bayes, and K-Nearest Neighbors.

## Data Preparation

### Binary Target Variable
Created a binary variable `mpg01` that indicates whether a car's MPG is above or below the median:
- `mpg01 = 1` if MPG > median
- `mpg01 = 0` if MPG ≤ median

The median MPG value in the Auto dataset serves as the threshold for classification.

## Exploratory Data Analysis

Visual exploration revealed strong relationships between `mpg01` and several predictor variables:

![Horsepower vs MPG01](plots/boxplot_horsepower.png)
**Figure 1:** Boxplot showing horsepower distribution by MPG category. Lower horsepower is associated with high MPG.

![Weight vs MPG01](plots/boxplot_weight.png)
**Figure 2:** Boxplot showing weight distribution by MPG category. Lighter vehicles tend to have higher MPG.

![Horsepower vs Weight Scatter](plots/scatter_horsepower_weight.png)
**Figure 3:** Scatter plot of horsepower vs weight colored by MPG category, showing clear separation between classes.

## Methodology

### Train-Test Split
The dataset was split into training (70%) and test (30%) sets to evaluate model performance on unseen data.

### Models Evaluated
1. **Linear Discriminant Analysis (LDA)** - Assumes equal covariance matrices across classes
2. **Quadratic Discriminant Analysis (QDA)** - Allows different covariance matrices per class
3. **Logistic Regression** - Models log-odds as linear function of predictors
4. **Naive Bayes** - Assumes feature independence within each class
5. **K-Nearest Neighbors (KNN)** - Non-parametric method tested with K values from 1 to 20

### Predictor Variables
Based on exploratory analysis, the following variables were used:
- `horsepower`
- `weight`
- `displacement`
- `acceleration`

## Results

### Test Error Rates

| Model | Test Accuracy | Test Error Rate |
|-------|---------------|-----------------|
| Linear Discriminant Analysis | 83.9% | 16.1% |
| Quadratic Discriminant Analysis | 83.1% | 16.9% |
| Logistic Regression | 84.7% | 15.3% |
| Naive Bayes | 83.1% | 16.9% |
| KNN (K=3) | 86.4% | 13.6% |

### KNN Performance Across K Values

![KNN Accuracy vs K](plots/knn_accuracy_vs_k.png)
**Figure 4:** Test accuracy for different K values in KNN. The optimal K value balances bias and variance.

The best performing K value was **K=3** with a test accuracy of **86.4%**.

## Conclusions

**Best Performing Model:** K-Nearest Neighbors (K=3) achieved the lowest test error rate of 13.6%, corresponding to the highest test accuracy of 86.4%.

**Key Findings:**
- KNN with K=3 outperformed all parametric methods (LDA, QDA, Logistic Regression, Naive Bayes), suggesting that non-linear decision boundaries with local smoothing are most appropriate for this data
- Logistic Regression achieved the best performance among parametric models with 84.7% accuracy
- QDA and Naive Bayes tied at 83.1% accuracy, both slightly underperforming LDA at 83.9%
- The close performance between QDA and LDA suggests that the assumption of equal covariance matrices is approximately reasonable for this dataset
- Variables `horsepower` and `weight` are strong predictors of MPG category, as evidenced by the clear visual separation in exploratory plots

**Model Rankings:**
1. KNN (K=3): 86.4%
2. Logistic Regression: 84.7%
3. LDA: 83.9%
4. QDA & Naive Bayes: 83.1% (tied)

**Practical Implications:**
The classification model can effectively predict whether a vehicle will achieve high or low fuel efficiency based on its physical characteristics, with over 86% accuracy. This could be useful for consumer decision-making when purchasing vehicles and for automotive manufacturers in designing fuel-efficient cars.