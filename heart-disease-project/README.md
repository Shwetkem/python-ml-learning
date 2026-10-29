# Predicting Heart Disease with Machine Learning

A binary classification project that predicts whether a patient has heart disease from 13 clinical measurements such as age, chest pain type, blood pressure and cholesterol. Built in Python with scikit-learn.

**Best model:** tuned Logistic Regression (`GridSearchCV`) with **88.5% test accuracy**, **92% cross-validated recall** and a test ROC/AUC curve.

## Problem

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

The goal set at the start was 95% accuracy on this proof of concept. The best result reached was 88.5%, so the project lists further experiments (see [Next steps](#next-steps)).

## Dataset

- File: `heart disease classification dataset.csv`
- Source: the Cleveland data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), also available on [Kaggle](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- 303 patients, 13 features and a `target` column (1 = heart disease, 0 = none)
- Class balance: 165 with heart disease, 138 without

| Feature | Meaning |
|---------|---------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0 to 3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG result (0 to 2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0 to 3) coloured by fluoroscopy |
| `thal` | Thallium stress test result |

## Workflow

1. **Cleaning:** dropped the extra `Unnamed: 0` index column. Converted `sex` and `target` from text to 0/1. Filled missing values in `trestbps` (4), `chol` (1) and `thalach` (5) with each column's mean.
2. **Exploratory analysis:** bar charts and crosstabs of heart disease against sex and chest pain type, a scatter plot of age against maximum heart rate, an age histogram and a correlation heatmap.
3. **Train/test split:** 80/20 (242 training rows, 61 test rows) with a fixed random seed of 42.
4. **Baseline models:** Logistic Regression, K-Nearest Neighbours and Random Forest.
5. **Hyperparameter tuning:** manual loop for KNN, `RandomizedSearchCV` for Logistic Regression and Random Forest, and `GridSearchCV` for Logistic Regression.
6. **Evaluation:** ROC curve and AUC, confusion matrix, classification report, and 5-fold cross-validated accuracy, precision, recall and F1.
7. **Feature importance:** read from the Logistic Regression coefficients.
8. **Saving:** exported the tuned model with `pickle` and checked that the reloaded model gives the same score.

## Results

### Model comparison (test accuracy)

| Model | Baseline | After tuning |
|-------|----------|--------------|
| K-Nearest Neighbours | 63.9% | 72.1% (best `n_neighbors` from 1 to 20) |
| Random Forest | 85.2% | 86.9% (`RandomizedSearchCV`) |
| Logistic Regression | 86.9% | 83.6% (`RandomizedSearchCV`), **88.5%** (`GridSearchCV`) |

Logistic Regression scored best at baseline, so it was tuned further with `GridSearchCV`. The best settings were `C = 0.204` with the `liblinear` solver.

### Tuned Logistic Regression in detail

Confusion matrix on the 61 test patients:

| | Predicted no disease | Predicted disease |
|---|---|---|
| **Actually no disease** | 25 | 4 |
| **Actually disease** | 3 | 29 |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| No disease (0) | 0.89 | 0.86 | 0.88 |
| Disease (1) | 0.88 | 0.91 | 0.89 |

Cross-validated metrics (5 folds, whole dataset):

| Metric | Score |
|--------|-------|
| Accuracy | 84.1% |
| Precision | 81.6% |
| Recall | 92.1% |
| F1 | 86.5% |

The cross-validated recall is the highest of the four, which suits a medical screening task where missing a patient with heart disease costs more than a false alarm.

### Feature importance

Logistic Regression coefficients, largest effects first:

| Pushes towards heart disease | Pushes away |
|------------------------------|-------------|
| `cp` (+0.66), `slope` (+0.47), `restecg` (+0.32) | `sex` (-0.84), `thal` (-0.67), `ca` (-0.65), `exang` (-0.60), `oldpeak` (-0.57) |

The features were not scaled, so the sizes of coefficients for features with large numeric ranges (`chol`, `trestbps`) are not directly comparable with the rest.

## Files

| File | What it is |
|------|------------|
| `heart-disease-ML.ipynb` | Full notebook: analysis, models, tuning and evaluation |
| `heart disease classification dataset.csv` | The dataset |
| `gs_log_reg_1.pkl` | Saved tuned model (the fitted `GridSearchCV` object) |

## How to run

```bash
git clone <this repository's URL>
cd python-ml-learning/heart-disease-project
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open `heart-disease-ML.ipynb` and run the cells from top to bottom.

To use the saved model:

```python
import pickle

with open("gs_log_reg_1.pkl", "rb") as f:
    model = pickle.load(f)

model.predict(X_test)  # X_test needs the same columns and encoding as in the notebook
```

## Next steps

- Collect more data.
- Try stronger models such as CatBoost or XGBoost.
- Improve the current models further, for example with feature scaling and by trying other hyperparameters.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, Jupyter Notebook
