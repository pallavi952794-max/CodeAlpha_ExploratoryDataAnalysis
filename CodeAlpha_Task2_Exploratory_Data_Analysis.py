"""
CodeAlpha Data Analytics Internship
Task 2: Exploratory Data Analysis (EDA)
Project: CodeAlpha_ExploratoryDataAnalysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("tab10")

def load_or_create_dataset():
    np.random.seed(42)
    n = 1000
    customer_ids = [f"CUST_{1000 + i}" for i in range(n)]
    ages = np.random.randint(18, 70, size=n)
    genders = np.random.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    categories = np.random.choice(["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty"], size=n)
    annual_income = np.random.normal(65000, 20000, size=n).clip(20000, 150000)
    purchase_amount = np.random.exponential(scale=120, size=n) + 15
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.08, 0.12, 0.25, 0.35, 0.20])
    churn = np.where((satisfaction_score <= 2) | (purchase_amount < 30), 
                     np.random.choice([0, 1], size=n, p=[0.3, 0.7]), 
                     np.random.choice([0, 1], size=n, p=[0.85, 0.15]))

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Category": categories,
        "AnnualIncome": np.round(annual_income, 2),
        "PurchaseAmount": np.round(purchase_amount, 2),
        "SatisfactionScore": satisfaction_score,
        "Churn": churn
    })

    missing_indices = np.random.choice(n, size=25, replace=False)
    df.loc[missing_indices, "AnnualIncome"] = np.nan
    return df

def perform_eda(df):
    print("="*60)
    print("PHASE 1: RESEARCH QUESTIONS & OBJECTIVES")
    print("="*60)
    print("1. Demographic profile of the customer base.")
    print("2. Spending differences across product categories.")
    print("3. Gender-based spending equality hypothesis test.")
    print("4. Core drivers of customer churn.")
    
    print("\n" + "="*60)
    print("PHASE 2: DATA STRUCTURE & PROFILING")
    print("="*60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe().T)
    
    print("\n" + "="*60)
    print("PHASE 3: DATA CLEANING & PREPROCESSING")
    print("="*60)
    print("Missing Values Detected:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    median_income = df["AnnualIncome"].median()
    df["AnnualIncome"].fillna(median_income, inplace=True)
    print(f"[+] Imputed 'AnnualIncome' with median: ${median_income:,.2f}")
    
    q25, q75 = np.percentile(df["PurchaseAmount"], [25, 75])
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    outliers = df[df["PurchaseAmount"] > upper_bound]
    print(f"[+] Outliers in 'PurchaseAmount' (above ${upper_bound:.2f}): {len(outliers)}")

    print("\n" + "="*60)
    print("PHASE 4: STATISTICAL HYPOTHESIS TESTING")
    print("="*60)
    male_spend = df[df["Gender"] == "Male"]["PurchaseAmount"]
    female_spend = df[df["Gender"] == "Female"]["PurchaseAmount"]
    t_stat, p_val = stats.ttest_ind(male_spend, female_spend, equal_var=False)
    print(f"Two-sample t-test: t={t_stat:.4f}, p-value={p_val:.4f}")
    if p_val < 0.05:
        print("[*] Result: Reject Null Hypothesis (Significant spending difference).")
    else:
        print("[*] Result: Fail to Reject Null Hypothesis (No significant spending difference).")

    print("\n" + "="*60)
    print("PHASE 5: GENERATING VISUALIZATIONS")
    print("="*60)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CodeAlpha Task 2 - Exploratory Data Analysis Dashboard", fontsize=16, fontweight='bold')
    
    sns.histplot(df["Age"], kde=True, ax=axes[0, 0], color="#2b5c8f", bins=20)
    axes[0, 0].set_title("Customer Age Distribution", fontweight='bold')
    
    sns.boxplot(x="Category", y="PurchaseAmount", data=df, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Purchase Amount by Product Category", fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=20)
    
    num_cols = ["Age", "AnnualIncome", "PurchaseAmount", "SatisfactionScore", "Churn"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 0])
    axes[1, 0].set_title("Correlation Heatmap", fontweight='bold')
    
    churn_rate = df.groupby("SatisfactionScore")["Churn"].mean().reset_index()
    sns.barplot(x="SatisfactionScore", y="Churn", data=churn_rate, ax=axes[1, 1], palette="Blues_d")
    axes[1, 1].set_title("Churn Rate by Satisfaction Score", fontweight='bold')
    axes[1, 1].set_ylabel("Churn Probability")
    
    plt.tight_layout()
    output_plot = "eda_summary_dashboard.png"
    plt.savefig(output_plot, dpi=300)
    print(f"[+] Saved visualization dashboard to '{output_plot}'!")
    
    clean_csv = "cleaned_eda_retail_data.csv"
    df.to_csv(clean_csv, index=False)
    print(f"[+] Exported cleaned dataset to '{clean_csv}'!")

if __name__ == "__main__":
    df = load_or_create_dataset()
    perform_eda(df)
