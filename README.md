# Task 2: Exploratory Data Analysis (EDA)

## Project Overview
This project performs an exploratory data analysis (EDA) pipeline on retail customer transaction data. The goal is to uncover behavioral patterns, profile demographic segments, and identify drivers of customer churn.

## Key Steps & Methodology
1. **Data Inspection:** Evaluated shape, data types, and non-null values.
2. **Missing Data Handling:** Handled missing income entries through median imputation.
3. **Outlier Detection:** Utilized Interquartile Range (IQR) bounds on transaction amounts.
4. **Hypothesis Testing:** Conducted two-sample independent t-testing to assess spending variances across customer genders.
5. **Multi-variable Visualization:** Generated a comprehensive 4-panel analytical dashboard (`eda_summary_dashboard.png`).

## Statistical Findings
- **Gender Spending:** The difference in purchase amounts between male and female customers yielded p > 0.05, confirming no statistically significant gender bias in spending.
- **Churn Driver:** Customer satisfaction rating is strongly inversely correlated with churn rate (Score 1 & 2 exhibit >65% churn risk).

## Execution
```bash
pip install pandas numpy matplotlib seaborn scipy
python CodeAlpha_Task2_Exploratory_Data_Analysis.py
```
