# 📉 Telco Customer Churn Prediction | LightGBM Approach

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling Strategy](#modeling-strategy)
6. [Results & Evaluation](#results--evaluation)
7. [Submission](#submission)
8. [Conclusion](#conclusion)

---

## 🚀 Introduction
Customer churn is a critical metric for telecommunications companies. Acquiring a new customer is significantly more expensive than retaining an existing one. This notebook presents a machine learning pipeline designed to predict customer churn based on demographic data, service usage, and payment information.

This project is built for the **Kaggle Playground Series - Season 6 Episode 3** competition. The goal is to classify whether a customer will churn (`Yes`) or stay (`No`) using a **LightGBM** classifier.

---

## 📊 Dataset Overview
The dataset contains information about telco customers including:
- **Demographics:** Gender, Senior Citizen status, Partner, Dependents.
- **Account Information:** Tenure, Contract type, Payment method, Paperless billing.
- **Services:** Phone service, Multiple lines, Internet service (DSL/Fiber), Online security, Backup, Device protection, Tech support, Streaming TV/Movies.
- **Financials:** Monthly Charges, Total Charges.
- **Target Variable:** `Churn` (Yes/No).

**Data Source:** [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)

**Kaggle Notebook:** [Kaggle Playground Series S6E3](https://www.kaggle.com/code/omkarkashid1168/kaggles6e3)
---

## 🔍 Exploratory Data Analysis (EDA)
Before modeling, extensive EDA was conducted to understand feature distributions and their relationship with the target variable.

### Key Findings:
1.  **Tenure vs. Churn:** There is a strong negative correlation between tenure and churn. Customers with low tenure (<12 months) are significantly more likely to churn compared to long-term customers.
2.  **Contract Type:** Month-to-month contracts exhibit the highest churn rates, while Two-year contracts show the highest retention.
3.  **Payment Method:** Customers paying via **Electronic Check** have a disproportionately higher churn rate compared to automatic payment methods (Bank transfer/Credit card).
4.  **Internet Service:** Fiber optic users show higher churn rates compared to DSL users, potentially due to higher monthly charges associated with fiber plans.
5.  **Correlations:** `MonthlyCharges` and `TotalCharges` are highly correlated with `tenure`. `InternetService` type is strongly linked to additional services (Online Security, Tech Support, etc.).

*(Visualizations included in the notebook: Heatmaps, Boxplots, Countplots, and Distribution histograms)*

---

## 🛠 Data Preprocessing
To prepare the data for the LightGBM model, the following preprocessing steps were applied:

### 1. Binary Encoding
Categorical columns with two unique values were mapped to integers (0/1):
- `gender`: Male → 1, Female → 0
- `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`: Yes → 1, No → 0
- `Churn`: Yes → 1, No → 0

### 2. Feature Engineering
- **Multiple Lines:** "No phone service" and "No" were grouped as 0, "Yes" as 1.
- **Internet Service Add-ons:** Columns like `OnlineSecurity`, `OnlineBackup`, etc., contained "No internet service". These were standardized to "No" before binary encoding (No → 0, Yes → 1).

### 3. One-Hot Encoding
Nominal categorical variables with more than two categories were one-hot encoded using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity:
- `InternetService`
- `Contract`
- `PaymentMethod`

### 4. Data Splitting
- The data was split into training and validation sets using `train_test_split`.
- **Split Ratio:** 80% Train, 20% Test.
- **Stratification:** Enabled (`stratify=y`) to maintain the class distribution of the target variable in both sets.

---

## 🤖 Modeling Strategy
### Algorithm: LightGBM Classifier
LightGBM was chosen for its efficiency, speed, and high performance on tabular data classification tasks.

### Hyperparameters
```python
model = lgb(
    class_weight='balanced',  # Handles class imbalance
    max_depth=12,             # Controls model complexity
    learning_rate=0.02,       # Slow learning rate for better convergence
    n_estimators=201          # Number of boosting rounds
)
```
- **Class Weighting:** Since the dataset is imbalanced (more non-churn customers than churn customers), `class_weight='balanced'` was used to penalize misclassification of the minority class more heavily.

---

## 📈 Results & Evaluation
The model was evaluated on the hold-out test set using classification metrics.

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **No Churn (0)** | 0.96 | 0.79 | 0.87 | 92,076 |
| **Churn (1)** | 0.55 | 0.88 | 0.68 | 26,763 |
| **Accuracy** | | | **0.81** | 118,839 |

### Analysis
- **Accuracy:** The model achieves an overall accuracy of **81%**.
- **Recall (Churn):** The model successfully identifies **88%** of the churned customers. This is crucial for business intervention strategies.
- **Precision (Churn):** The precision for the churn class is lower (55%), indicating some false positives. However, in churn prediction, it is often preferable to flag a loyal customer as at-risk (False Positive) than to miss a churned customer (False Negative).

---

## 📤 Submission
The preprocessing pipeline applied to the training data was replicated on the test set to ensure consistency.
1. Test data was cleaned and encoded using the same mapping rules.
2. Columns were reindexed to match the training feature set.
3. Predictions were generated and mapped back to 'Yes'/'No' labels.
4. Results were saved to `submission.csv`.

```python
submission = pd.DataFrame({
    'id': test['id'], 
    'Churn': y_pred_test
})
submission['Churn'] = submission['Churn'].map({0: 'No', 1: 'Yes'})
submission.to_csv('submission.csv', index=False)
```

---

## 🏁 Conclusion
This notebook demonstrates a robust pipeline for customer churn prediction. By leveraging LightGBM with class balancing and thorough preprocessing, we achieved a strong recall score for the churn class.

### Future Improvements
- **Hyperparameter Tuning:** Utilize `Optuna` or `GridSearchCV` to fine-tune LightGBM parameters further.
- **Feature Selection:** Use feature importance scores to remove redundant features and reduce overfitting.
- **Ensembling:** Combine predictions with other models like XGBoost or CatBoost for potential performance gains.

---

## 🙏 Acknowledgments
- **Kaggle:** For hosting the Playground Series competition.
- **LightGBM Developers:** For the efficient gradient boosting framework.

*If you find this notebook useful, please consider upvoting!* 🚀
