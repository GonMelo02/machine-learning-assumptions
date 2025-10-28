# Exploring Machine Learning Assumptions and Model Behavior
  
The main objective of this project is to investigate how different machine learning algorithms behave under varying data characteristics, such as class balance, noise, and decision boundary complexity.  

The project explores both the **assumptions** and the **limitations** of classical ML methods through synthetic data experiments, enabling visual and quantitative understanding of how models generalize under different conditions.

---

## Objectives

- Generate synthetic datasets that illustrate the assumptions and performance boundaries of common machine learning algorithms.
- Explore how dataset properties (e.g., class overlap, noise, nonlinearity) affect each model.
- Study bias–variance decomposition and its impact on model capacity and generalization.
- Compare ensemble methods (Bagging, Random Forest, AdaBoost) in terms of learning curves, stability, and accuracy.

---

## Methods

### Models Investigated
- **Logistic Regression**
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**
- **Decision Trees** (with and without pruning)
- **Support Vector Machines (Linear and RBF)**
- **Ensemble Methods**: Bagging, Random Forest, AdaBoost

### Experimental Design
- Artificial 2D datasets were generated to control for:
  - Class balance and number of samples  
  - Degree of noise and overlap between classes  
  - Shape of the class boundaries (linear vs. nonlinear)  
- Each model was evaluated under conditions favorable and unfavorable to its assumptions.

---

## Bias–Variance Decomposition

Decision Trees were used to investigate the **bias–variance tradeoff**:  
- Increasing noise or pruning levels altered the model’s capacity.  
- Bias, variance, and total error were measured across different configurations to illustrate overfitting and underfitting behavior.

---

## Ensemble Analysis

Three ensemble methods were compared:
- **Bagging:** variance reduction via bootstrap aggregation  
- **Random Forest:** improved decorrelation of trees  
- **AdaBoost:** adaptive reweighting of misclassified instances  

Their performance was compared through:
- Out-of-Bag (OOB) error curves  
- Learning curves over number of estimators  
- Cross-validation results  

---

## Technologies Used

- **Python**
- `scikit-learn`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scipy`
- `Jupyter Notebook`

