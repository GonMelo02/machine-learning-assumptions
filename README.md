# ML Assumptions & Bias-Variance Exploration

Systematic experimental study of 7 classical ML algorithms 
using synthetic 2D datasets, covering method assumptions, 
bias-variance decomposition, and ensemble comparison.

## Ensemble Results (Make Moons, noise=0.2, 5-fold CV, 200 estimators)

| Method | CV Accuracy |
|--------|-------------|
| AdaBoost | ~0.850 |
| Random Forest | ~0.841 |
| Bagging | ~0.830 |

## Experiments

**1. Method Assumptions**
Generated targeted synthetic datasets to isolate conditions 
where each model dominates: linearly separable (LR/LDA/SVM Linear), 
Gaussian with unequal covariance (QDA), nonlinear boundaries 
(SVM RBF, Decision Tree).

**2. Bias-Variance Decomposition**
Decision Trees evaluated across ccp_alpha values — measured 
bias, variance, and total error decomposition under varying 
noise levels to illustrate underfitting/overfitting tradeoff.

**3. Ensemble Comparison**
Bagging, Random Forest, AdaBoost compared via OOB error curves 
and cross-validation. Random Forest most robust; AdaBoost 
sensitive to noise.

## Stack
Python · scikit-learn · numpy · matplotlib · seaborn

## Run locally
pip install -r requirements.txt
jupyter notebook machine_learning_experiments.ipynb
