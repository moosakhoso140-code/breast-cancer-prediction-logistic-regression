# breast-cancer-prediction-logistic-regression
Predicting whether a tumor is malignant using Logistic Regression in Python (scikit-learn), with a focus on recall


# Breast Cancer Prediction using Logistic Regression

A beginner-level machine learning project that predicts whether a tumor is malignant or not, using Logistic Regression.

## Problem
Given measurements of a tumor, predict whether it is **malignant (1)** or **not malignant (0)**.
In medical problems, missing a real cancer case is more serious than a false alarm, so I focused on **recall** and not only accuracy.

## Dataset
- Source: [ Kaggle dataset ]
- Target column: `Class` (0 = not malignant, 1 = malignant)
- Class balance in the training set: 364 rows of class 0 and 181 rows of class 1 (about 67% / 33%)

## Tools Used
- Python
- pandas
- scikit-learn
- JupyterLab

## Steps
1. Loaded the data with pandas and checked for missing values
2. Separated the target (`Class`) from the features so the target does not leak into X
3. Split the data into 80% train and 20% test (`random_state=42`)
4. Built a pipeline: `StandardScaler` followed by `LogisticRegression`
5. Evaluated the model with accuracy, precision, recall, F1 and a confusion matrix
6. Compared train and test accuracy to check for overfitting

## Results

| Metric | Value |
|---|---|
| Test accuracy | 0.956 |
| Precision | 0.981 |
| Recall | 0.912 |
| F1-score | 0.945 |
| Train accuracy | 0.971 |

Confusion matrix on the test set (137 rows):

|  | Predicted: not malignant | Predicted: malignant |
|---|---|---|
| **Actual: not malignant** | 79 | 1 |
| **Actual: malignant** | 5 | 52 |

## What I Learned
- **Accuracy alone is not enough.** The model has 0.98 precision, but recall is 0.91, which means 5 malignant cases were missed. In a medical setting these false negatives matter most.
- **Overfitting check:** train accuracy (0.971) and test accuracy (0.956) are close, so the model is not overfitting.
- **Class balance:** the data is slightly imbalanced (67/33). A model that always predicts "not malignant" would get about 67% accuracy, so 95.6% is a real improvement.
- **Scaling:** I used a pipeline so the scaler learns only from the training data and no test information leaks in.
- **Threshold:** the default probability threshold is 0.5. Lowering it can improve recall at the cost of precision.

## Limitations
- The test set is small (137 rows), so one or two rows change the scores noticeably.
- Only Logistic Regression was tried.

## Future Work
- Tune the probability threshold to improve recall
- Use cross-validation for more reliable scores
- Compare with Decision Tree and Random Forest


## Author
[Muhammad Moosa khan] | [https://www.linkedin.com/in/muhammad-moosa-khan-b96563261]
