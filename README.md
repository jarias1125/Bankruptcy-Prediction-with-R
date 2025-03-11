# Finance & Data Analysis with R

## Project Overview
This project applies data science and machine learning techniques using R to analyze financial data. The study focuses on **credit scoring analysis** using a dataset from the **Taiwan Economic Journal (1999)**. The objective is to explore data patterns, build predictive models, and evaluate their effectiveness in financial decision-making.

## Dataset
- **Source:** Taiwan Economic Journal (1999)
- **Sample Size:** 30,000 customer records
- **Target Variable:** Credit Default (Binary: 1 = Default, 0 = No Default)
- **Data Type:** Structured tabular data
- **Key Features:**
  - **Demographic Information:** Age, Gender, Marital Status, Education Level
  - **Financial Behavior:** Credit Limit, Outstanding Bills, Past Payment History
  - **Repayment Details:** Amount paid in past billing cycles, delayed payments
- **Data Distribution:** The dataset is imbalanced, with approximately **22% of customers defaulting** on credit payments and **78% repaying on time**. SMOTE and undersampling techniques were applied to address this imbalance.

## Features
- Data cleaning and preprocessing
- Exploratory data analysis (EDA) and visualization
- Feature engineering and selection
- Predictive analytics using machine learning models
- Performance evaluation and result interpretation

## Technologies Used
- **R Programming Language**
- **Libraries:**
  - Data Manipulation: `tidyverse`, `dplyr`, `reshape2`
  - Machine Learning: `caret`, `catboost`, `tidymodels`
  - Feature Engineering: `smotefamily`, `themis`, `recipes`
  - Model Evaluation: `pROC`, `DMwR2`

## Results and Insights
### Predictive Model Performance
- **Logistic Regression:**
  - Accuracy = **91.89%**
  - Balanced Accuracy = **74.00%**
  - AUC = **0.88**
- **CatBoost (Tuned):**
  - Sensitivity (Recall for Bankrupt Companies) = **0.97**
  - F1-Score = **0.98**
  - AUC = **0.93**
  - Specificity (Correctly Identifying Non-Bankrupt Companies) = **0.51**

The **tuned CatBoost model** demonstrates exceptional performance in identifying bankrupt companies, achieving a very high **sensitivity of 0.97**, making it highly effective at predicting the minority class (bankrupt companies). The **F1-Score of 0.98** further confirms its strong balance between precision and recall. With an **AUC of 0.93**, the model also shows excellent discriminatory power. However, the **specificity (0.51)** suggests it sacrifices some ability to correctly identify non-bankrupt companies for better detection of bankrupt ones.

### Feature Importance Analysis
The feature importance graph highlights the **top 15 predictors of bankruptcy** in the **tuned CatBoost model**. The most influential features include:

1. **Fixed Assets Turnover Frequency** – Key determinant of bankruptcy, indicating how effectively fixed assets are utilized.
2. **Inventory Turnover Rate (times)** – Reflects inventory management efficiency and conversion into revenue.
3. **Borrowing Dependency** – Highlights reliance on external borrowing as a potential risk factor.
4. **Operating Expense Rate** – Shows the proportion of operating expenses to revenue, critical for profitability.
5. **Interest-bearing Debt Interest Rate** – Indicates the cost of debt financing.

These findings suggest that **operational efficiency, financial structure, and borrowing practices significantly impact bankruptcy likelihood**.

### Conclusion
The project successfully addressed the problem of **predicting bankruptcy** using multiple machine learning models and data balancing techniques like **SMOTE**. Logistic regression served as a **baseline model**, while **CatBoost (both default and tuned)** significantly outperformed other models in terms of **AUC and accuracy**. However, the **tuned CatBoost model** demonstrated higher **precision and sensitivity** in identifying bankrupt companies, making it a strong candidate for **real-world financial risk assessment applications**.

## Contact
For questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/).
