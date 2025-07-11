# Predicting Employee Attrition using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--Learn%20%7C%20Pandas%20%7C%20SHAP-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

### Project Overview

This project presents an end-to-end machine learning solution for predicting employee attrition. By leveraging the IBM HR Analytics dataset, the primary goal is to build a robust model that can proactively identify employees who are at a high risk of leaving the company. The insights derived from this model can empower HR departments to implement targeted retention strategies, ultimately reducing turnover costs and preserving valuable talent.

This repository showcases a complete data science workflow, from initial data exploration and cleaning to advanced feature engineering, systematic model evaluation, and in-depth, interpretable insights.

### Key Features & Technical Highlights

- **Systematic Model Evaluation:** A custom `ModelTrainer` class was developed to benchmark over 20 classification algorithms (including XGBoost, LightGBM, and CatBoost) in a structured and repeatable manner. This framework automates the entire training, evaluation, and tuning process.
- **Best-Practice Pipelining:** Implemented `scikit-learn` and `imblearn` pipelines to ensure that data preprocessing steps, especially resampling for class imbalance, were correctly applied within each cross-validation fold, preventing data leakage.
- **Advanced Imbalance Handling:** Utilized the SMOTE-Tomek (a hybrid over-sampling and under-sampling) technique to address the significant class imbalance (16% attrition rate) in the dataset, leading to more reliable model performance.
- **Rigorous Hyperparameter Tuning:** Employed `RandomizedSearchCV` for efficient and comprehensive hyperparameter optimization across multiple models, focusing on maximizing the ROC-AUC score.
- **In-Depth Model Interpretation:** Went beyond performance metrics to explain the "why" behind the model's predictions using:
  - **Permutation Importance:** To identify the most influential features on the hold-out test set.
  - **SHAP (SHapley Additive exPlanations):** To visualize the magnitude and direction of each feature's impact on individual predictions, translating the complex MLP model into actionable business insights.
- **Statistical Validation:** Conducted McNemar's test and a paired t-test to statistically compare the top-performing models, ensuring that performance differences were significant and not due to random chance.

### Tech Stack

- **Core Libraries:** `Python`, `Pandas`, `NumPy`, `Scikit-learn`, `Imbalanced-learn`
- **Gradient Boosting:** `XGBoost`, `LightGBM`, `CatBoost`
- **Model Interpretation:** `SHAP`, `Matplotlib`, `Seaborn`
- **Statistical Analysis:** `Statsmodels`, `SciPy`

### Project Workflow

1.  **Exploratory Data Analysis (EDA):** Investigated the dataset to understand feature distributions, correlations, and the extent of class imbalance.
2.  **Data Preprocessing & Feature Engineering:**
    - Handled categorical and numerical features, removing uninformative columns.
    - Created new, insightful features from existing data, such as pay-rate ratios and tenure-based metrics.
    * Applied log transformations to skewed numerical features and scaled all numerical data using `StandardScaler`.
3.  **Modeling & Evaluation:**
    - Utilized the custom `ModelTrainer` to train and evaluate a wide array of models on a baseline level.
    - Key evaluation metrics included ROC-AUC and PR-AUC, which are well-suited for imbalanced classification, alongside Precision, Recall, and F1-Score.
4.  **Hyperparameter Tuning:**
    - Performed an automated `RandomizedSearchCV` for the most promising models to find the optimal hyperparameter configurations.
    - The Multi-layer Perceptron (MLP) emerged as the top-performing model after tuning, with a ROC-AUC of **0.819**.
5.  **Model Interpretation & Insight Generation:**
    - Analyzed the final MLP model's confusion matrix to understand its performance in a business context (i.e., the cost of false negatives vs. false positives).
    - Used Permutation Importance and SHAP to uncover the key drivers of attrition.

### Key Insights from the Best Model (MLP)

The model interpretation revealed clear, actionable patterns for why employees leave. These insights can directly inform retention policies.

- **Overtime is the #1 Predictor:** `OverTime_Yes` was overwhelmingly the most significant factor driving attrition. Employees working overtime are far more likely to be predicted as leaving.
- **Travel & Compensation Matter:** `BusinessTravel_Travel_Frequently` was the second-largest contributor to attrition risk. Furthermore, lower `MonthlyIncome` was a strong predictor of leaving.
- **Tenure with Manager is Crucial:** A shorter tenure with the current manager (`YearsWithCurrManager`) significantly increased the predicted risk of attrition, highlighting the importance of the manager-employee relationship in the early stages.
- **Job Role is a Key Factor:** Specific job roles, such as `Sales Executive` and `Laboratory Technician`, were identified by the model as having a higher propensity for attrition.

### How to Use This Repository

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    _(Note: You may want to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project environment.)_

3.  **Explore the project:**
    - `binary-classification-hr-attrition.pdf`: A comprehensive PDF report of the Jupyter Notebook containing the full analysis, code, and visualizations.
    - `models.py`: The custom `ModelTrainer` class used for systematic modeling.
    - `WA_Fn-UseC_-HR-Employee-Attrition.csv`: The raw dataset used for the project.
