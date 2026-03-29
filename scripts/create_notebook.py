import json

cells = []

def add_md(text):
    lines = text.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    })

def add_code(text):
    lines = text.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

add_md("""# E-Commerce Conversion Optimization & A/B Testing Dashboard

| Field | Details |
|:---|:---|
| **Project** | Causal A/B Testing Framework & Conversion Analysis |
| **Dataset** | Simulated E-Commerce Traffic (with embedded confounders) |
| **Author** | Sanman |
| **Date** | March 2026 |

---

## 1. Problem Statement

In the highly competitive e-commerce landscape, minimizing friction during the checkout process is critical for maximizing revenue. The current checkout flow on our platform has a high drop-off rate at the payment information step. The product team has proposed a new "One-Click Checkout" flow designed to streamline this process. 

**Why it matters:** Even a 1% increase in conversion rate can translate to millions of dollars in annualized revenue. However, blindly deploying the new feature without rigorous testing could lead to unforeseen negative impacts on average order value (AOV) or user retention.

**Expected Insights:** We need to determine if the "One-Click Checkout" (Variant) statistically outperforms the existing flow (Control) in terms of overall conversion rate, and understand if the effect varies across different user segments (e.g., Device Type, Customer Type).

## 2. Objectives

This analysis aims to achieve the following:
1. **Exploratory Goal:** Understand the baseline distributions of cart values, time spent, and conversion rates across different user demographics.
2. **Preprocessing Goal:** Clean and prepare the hypothetical dataset, handling missing values and scaling features appropriately.
3. **Statistical Goal 1:** Conduct a rigorous Frequentist A/B test (Two-Proportion Z-Test) to evaluate the primary conversion metric.
4. **Statistical Goal 2:** Apply a Logistic Regression model to control for confounding variables and estimate the causal impact of the new checkout flow.
5. **Business Goal:** Provide actionable recommendations based on the statistical findings, including segment-specific strategies.
""")

add_code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, ttest_ind, probplot
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Set aesthetic style for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

import warnings
warnings.filterwarnings('ignore')
""")

add_md("""## 3. Dataset Overview and Generation

Since no external dataset was provided, we will generate a robust hypothetical dataset representing an A/B test running for two weeks.

**Dataset Characteristics:**
* **`user_id`**: Unique identifier for the session.
* **`group`**: 'Control' (Old Checkout) or 'Variant' (One-Click Checkout).
* **`device_type`**: 'Mobile', 'Desktop', or 'Tablet'.
* **`customer_type`**: 'New' or 'Returning'.
* **`cart_value`**: The total dollar value of the items in the cart prior to checkout.
* **`time_spent_mins`**: Time spent on the site before entering the checkout flow.
* **`converted`**: Binary outcome (1 = completed purchase, 0 = abandoned).

**Assumptions & Potential Issues:**
* The assignment to Control/Variant is completely randomized (i.e., no assignment bias).
* The events are independent and identically distributed (i.i.d).
* We will intentionally introduce some missing data in `time_spent_mins` to demonstrate robust preprocessing techniques.
""")

add_code("""# Set random seed for reproducibility
np.random.seed(42)

# Generate basic attributes
n_users = 10000
user_ids = [f"U_{i:05d}" for i in range(1, n_users + 1)]
groups = np.random.choice(['Control', 'Variant'], size=n_users, p=[0.5, 0.5])
devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n_users, p=[0.6, 0.3, 0.1])
customer_types = np.random.choice(['New', 'Returning'], size=n_users, p=[0.4, 0.6])

# Generate Cart Value (Log-normal distribution to mimic real e-commerce data)
cart_values = np.random.lognormal(mean=3.5, sigma=0.8, size=n_users)
cart_values = np.round(np.clip(cart_values, 5, 500), 2)

# Generate Time Spent
time_spent = np.random.normal(loc=12, scale=5, size=n_users)
time_spent = np.round(np.clip(time_spent, 1, 60), 1)

# Introduce missing values in time_spent (approx 5%)
missing_idx = np.random.choice(n_users, size=int(n_users * 0.05), replace=False)
time_spent[missing_idx] = np.nan

# Generate Conversion Logits (Base probability + effects)
# Baseline conversion log-odds
logits = -1.5 
# Effect of Variant
logits += np.where(groups == 'Variant', 0.25, 0)
# Effect of Device (Mobile is harder to convert)
logits += np.where(devices == 'Mobile', -0.3, 0)
# Effect of Customer Type (Returning convert more)
logits += np.where(customer_types == 'Returning', 0.5, 0)
# Effect of Cart Value (Higher carts slightly less likely to convert directly)
logits -= cart_values * 0.001

# Calculate probabilities using sigmoid
probs = 1 / (1 + np.exp(-logits))
converted = np.random.binomial(n=1, p=probs)

# Create DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'group': groups,
    'device_type': devices,
    'customer_type': customer_types,
    'cart_value': cart_values,
    'time_spent_mins': time_spent,
    'converted': converted
})

display(df.head())
print(f"Dataset Shape: {df.shape}")
df.info()
""")

add_md("""**Interpretation:**
We have successfully generated a dataset of 10,000 unique user sessions. The `info()` output confirms the presence of missing values in `time_spent_mins`. The distributions and relationships built into the logits (e.g., returning users converting at higher rates, mobile devices showing lower conversion) will allow us to demonstrate both confounding and segment analysis later.

---

## 4. Data Preprocessing

Before proceeding to analysis, we must ensure the data is clean and ready for modeling.

### 4.1. Handling Missing Values
We observed missing values in the `time_spent_mins` column. Since time spent can vary heavily by `customer_type` and `device_type`, we will impute these missing values using the median of their respective subgroups to preserve the underlying distributions without being skewed by extreme outliers.
""")

add_code("""# Calculate percentage of missing values
missing_pct = df.isnull().mean() * 100
print("Missing Value Percentages:\\n", missing_pct[missing_pct > 0])

# Impute missing values with group medians
df['time_spent_mins'] = df.groupby(['device_type', 'customer_type'])['time_spent_mins'].transform(
    lambda x: x.fillna(x.median())
)

print(f"Remaining missing values: {df.isnull().sum().sum()}")
""")

add_md("""### 4.2. Outlier Detection
We will visually inspect `cart_value` for extreme outliers that might skew revenue-based metrics.
""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='cart_value', ax=axes[0])
axes[0].set_title('Boxplot of Cart Values')

# Log-transformed visualization
sns.histplot(np.log1p(df['cart_value']), kde=True, ax=axes[1])
axes[1].set_title('Log-transformed Distribution of Cart Values')
axes[1].set_xlabel('Log(Cart Value + 1)')

plt.tight_layout()
plt.show()
""")

add_md("""**Outlier Strategy:**
Our `cart_value` was generated with an upper clip of $500, simulating a natural cap or a pre-filtering step often done in e-commerce to separate wholesale from retail. The log-transformation shows a beautiful normal-like distribution, confirming that our values are highly right-skewed but manageable. We will not drop outliers as they represent valid (and highly valuable) purchases, but we will apply standard scaling to numerical features before running the logistic regression later to ensure convergence.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1. Overall Conversion & Group Balance
Let's first confirm that the A/B test groups are balanced and look at the aggregate conversion rates.
""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Group Balance
sns.countplot(data=df, x='group', palette=['#4c72b0', '#dd8452'], ax=axes[0])
axes[0].set_title('Traffic Split (Sample Size Check)')
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

# Baseline Conversion by Group
conv_rates = df.groupby('group')['converted'].mean().reset_index()
sns.barplot(data=conv_rates, x='group', y='converted', palette=['#4c72b0', '#dd8452'], ax=axes[1])
axes[1].set_title('Conversion Rate by Group')
axes[1].set_ylabel('Conversion Rate')
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()
""")

add_md("""**Interpretation:**
* **Sample Size Check:** The traffic split is practically 50/50, indicating our randomization mechanism worked and there is no Sample Ratio Mismatch (SRM). 
* **Conversion Rate:** In this seeded run, the Control group converts at roughly 19.4% while the Variant converts at about 24.2%. That is an absolute uplift of about 4.8 percentage points, which looks promising but still needs formal testing.

### 5.2. Segment Analysis (Device and Customer Type)
Let's see how conversion changes across different user dimensions.
""")

add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Conversion by Device Type
sns.barplot(data=df, x='device_type', y='converted', hue='group', ax=axes[0])
axes[0].set_title('Conversion Rate by Device Type')
axes[0].set_ylabel('Conversion Rate')

# Conversion by Customer Type
sns.barplot(data=df, x='customer_type', y='converted', hue='group', ax=axes[1])
axes[1].set_title('Conversion Rate by Customer Type')
axes[1].set_ylabel('Conversion Rate')

plt.tight_layout()
plt.show()
""")

add_md("""**Interpretation:**
* **Device Type:** Mobile has the lowest overall conversion rate out of the three device types. However, the Variant (One-Click Checkout) seems to have improved conversion across all devices, with a visible jump in Desktop as well.
* **Customer Type:** Returning customers convert at a significantly higher rate than New customers, which aligns with standard e-commerce behavior (higher trust, saved info). The Variant appears to benefit both segments positively.

### 5.3. Correlation Analysis
Let's visualize the linear relationships between our continuous variables and the binary target.
""")

add_code("""numeric_df = df[['cart_value', 'time_spent_mins', 'converted']]
corr_matrix = numeric_df.corr(method='spearman') # Spearman for non-linear/binary robust stats

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()
""")

add_md("""**Interpretation:**
* There is a weak negative correlation (-0.05) between `cart_value` and `converted`. Larger carts might require more deliberation, thus converting less frequently in a single session.
* The correlations are generally weak, which emphasizes the need for multivariate causal analysis rather than relying on simple linear bivariate relationships.

---

## 6. Statistical Hypothesis Testing

### 6.1. Business Decision Framing
The launch question is directional: we only want to ship the new checkout if it increases conversion.

**Hypotheses:**
* **$H_0$:** $p_{new} \\le p_{old}$ 
* **$H_1$:** $p_{new} > p_{old}$

**Statistical Test:**
* **Two-proportion z-test**
* **Significance level ($\\alpha$):** 0.05
* **Decision rule:** If the one-sided p-value is below 0.05, reject $H_0$ and support launch.
""")

add_code("""# Compute summary stats
control_conv = df[df['group'] == 'Control']['converted'].sum()
control_n = df[df['group'] == 'Control'].count()['user_id']
control_p = control_conv / control_n

variant_conv = df[df['group'] == 'Variant']['converted'].sum()
variant_n = df[df['group'] == 'Variant'].count()['user_id']
variant_p = variant_conv / variant_n

# Pooled probability
pooled_p = (control_conv + variant_conv) / (control_n + variant_n)

# Standard error
se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_n + 1/variant_n))

# Z-score and one-sided p-value for:
# H0: p_new <= p_old
# H1: p_new > p_old
z_stat = (variant_p - control_p) / se
p_value = 1 - norm.cdf(z_stat)

# 95% confidence interval for the uplift
se_diff = np.sqrt(
    (control_p * (1 - control_p) / control_n) +
    (variant_p * (1 - variant_p) / variant_n)
)
uplift = variant_p - control_p
ci_lower = uplift - 1.96 * se_diff
ci_upper = uplift + 1.96 * se_diff
relative_lift = uplift / control_p

print(f"Control Conversion: {control_p:.4f} ({control_conv}/{control_n})")
print(f"Variant Conversion: {variant_p:.4f} ({variant_conv}/{variant_n})")
print(f"Absolute Uplift:    {uplift:.4f}")
print(f"Relative Uplift:    {relative_lift:.2%}")
print("-" * 30)
print(f"Z-Statistic:        {z_stat:.4f}")
print(f"P-Value:            {p_value:.6f}")
print(f"95% CI (uplift):    [{ci_lower:.4f}, {ci_upper:.4f}]")

if p_value < 0.05:
    print("\\nBusiness Conclusion: We reject H0 -> statistically significant improvement -> launch is supported.")
else:
    print("\\nBusiness Conclusion: We fail to reject H0 -> no significant improvement -> do not launch.")
""")

add_md("""**Interpretation:**
The variant delivers a positive absolute uplift and the **p-value is shown explicitly**, so the decision is audit-friendly. Because the one-sided p-value is far below 0.05, this experiment provides evidence that the new checkout performs better than the old one.

### 6.2. Advanced Layer: Confidence Interval Visualization
P-values answer whether the observed lift is statistically surprising under the null. Confidence intervals add the missing business context: they show the plausible range of the uplift size, which helps assess whether the effect is merely significant or also meaningful.
""")

add_code("""ci_df = pd.DataFrame({
    'group': ['Control Conversion', 'Variant Conversion', 'Variant - Control Uplift'],
    'point_estimate': [control_p, variant_p, uplift],
    'ci_lower': [
        control_p - 1.96 * np.sqrt(control_p * (1 - control_p) / control_n),
        variant_p - 1.96 * np.sqrt(variant_p * (1 - variant_p) / variant_n),
        ci_lower
    ],
    'ci_upper': [
        control_p + 1.96 * np.sqrt(control_p * (1 - control_p) / control_n),
        variant_p + 1.96 * np.sqrt(variant_p * (1 - variant_p) / variant_n),
        ci_upper
    ]
})

plot_df = ci_df.iloc[::-1].reset_index(drop=True)

plt.figure(figsize=(10, 4))
plt.errorbar(
    x=plot_df['point_estimate'],
    y=plot_df['group'],
    xerr=[
        plot_df['point_estimate'] - plot_df['ci_lower'],
        plot_df['ci_upper'] - plot_df['point_estimate']
    ],
    fmt='o',
    color='#1f77b4',
    ecolor='#4c72b0',
    capsize=5
)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No uplift threshold')
plt.title('Confidence Intervals for Conversion Rates and Uplift')
plt.xlabel('Rate / Uplift')
plt.legend()
plt.tight_layout()
plt.show()

display(ci_df.style.format({
    'point_estimate': '{:.4f}',
    'ci_lower': '{:.4f}',
    'ci_upper': '{:.4f}'
}))
""")

add_md("""**Interpretation:**
The uplift confidence interval stays above zero, which strengthens the launch recommendation. This view is useful for executives because it frames uncertainty as a range of plausible business outcomes instead of a single test statistic.

---

## 7. Causal Inference via Logistic Regression

While the Z-Test tells us if the effect is significant, it doesn't account for complex confounding relationships (e.g., what if the Variant by chance got slightly more "Returning" customers, who convert easier?). 

To isolate the *Average Treatment Effect (ATE)* of the new checkout flow while controlling for covariates, we will fit a Logistic Regression model.

### 7.1. Feature Engineering and Scaling
""")

add_code("""# Create dummy variables for categoricals
df_model = df.copy()
df_model['is_variant'] = (df_model['group'] == 'Variant').astype(int)
df_model = pd.get_dummies(df_model, columns=['device_type', 'customer_type'], drop_first=True)

# Standardize continuous variables
scaler = StandardScaler()
df_model[['cart_value', 'time_spent_mins']] = scaler.fit_transform(df_model[['cart_value', 'time_spent_mins']])

# Define independent variables (X) and Dependent target (y)
features = ['is_variant', 'cart_value', 'time_spent_mins', 
            'device_type_Mobile', 'device_type_Tablet', 
            'customer_type_Returning']

X = sm.add_constant(df_model[features].astype(float))
y = df_model['converted']

# Fit Logistic Regression Model
logit_model = sm.Logit(y, X).fit(disp=False)
print(logit_model.summary())
""")

add_md("""### 7.2. Extracting the Treatment Effect (Odds Ratios)
Let's convert the log-odds coefficients into easily interpretable Odds Ratios.
""")

add_code("""# Calculate Odds Ratios and 95% Confidence Intervals
odds_ratios = pd.DataFrame({
    'Odds Ratio': np.exp(logit_model.params),
    'Lower CI': np.exp(logit_model.conf_int()[0]),
    'Upper CI': np.exp(logit_model.conf_int()[1]),
    'P-Value': logit_model.pvalues
})

# Filter out the constant for plotting
odds_ratios = odds_ratios.drop('const')
display(odds_ratios)

# Plotting the Odds Ratios
plt.figure(figsize=(10, 6))
plt.errorbar(x=odds_ratios['Odds Ratio'], y=odds_ratios.index, 
             xerr=[odds_ratios['Odds Ratio'] - odds_ratios['Lower CI'], 
                   odds_ratios['Upper CI'] - odds_ratios['Odds Ratio']], 
             fmt='o', color='purple', capsize=5, capthick=2)

plt.axvline(x=1, color='red', linestyle='--', label='No Effect (OR = 1)')
plt.title('Treatment & Covariate Effects (Odds Ratios with 95% CIs)')
plt.xlabel('Odds Ratio ( > 1 increases likelihood of conversion)')
plt.legend()
plt.tight_layout()
plt.show()
""")

add_md("""**Interpretation:**
* **`is_variant` (The Treatment Effect):** The odds ratio is approximately 1.33. This means that, *holding all other variables constant* (cart value, device, customer type), users in the Variant group are about **33% more likely to convert** than users in the Control group.
* **`customer_type_Returning`:** Returning customers are heavily predicted to convert (OR ~ 1.67), highlighting the importance of customer retention.
* **`device_type_Mobile`:** As expected based on our EDA, being on a mobile device decreases the odds of conversion compared to Desktop (the dropped reference baseline), as seen by an OR < 1.0.

**Why this matters over the z-test:** This proves that the observed uplift is not just a top-line artifact. Even after controlling for customer mix and device mix, the new checkout still improves conversion, which makes the rollout decision more defensible.

---

## 8. Financial Implication Simulation

To translate "Odds Ratios" and "Statistical Significance" into business language, let's simulate the annualized revenue impact if we roll this out to 100% of traffic.
* **Assumptions:** 
    * 100,000 visitors per month (1.2M / year).
    * Average Order Value (AOV) is derived from the simulated cart-value median, which is about $33.09 in this run.
""")

add_code("""annual_visitors = 1_200_000
aov = df['cart_value'].median() 

control_annual_conversion = control_p
variant_annual_conversion = variant_p

control_revenue = annual_visitors * control_annual_conversion * aov
variant_revenue = annual_visitors * variant_annual_conversion * aov
incremental_revenue = variant_revenue - control_revenue

print(f"Estimated AOV: ${aov:.2f}")
print(f"Projected Annual Revenue (Control Flow): ${control_revenue:,.2f}")
print(f"Projected Annual Revenue (Variant Flow): ${variant_revenue:,.2f}")
print(f"\\n💰 Projected Incremental Revenue: ${incremental_revenue:,.2f} per year")
""")

add_md("""**Business Takeaway:** If this lift holds at scale, rolling out the One-Click checkout is projected to generate about **$1.89 million** in incremental revenue annually by reducing checkout friction.

---

## 9. Decision Summary for Stakeholders

This is the version a business analyst would present in a launch review:
* The new checkout increased conversion rate versus the old flow.
* The formal test was a **two-proportion z-test** aligned to the directional launch question.
* The **p-value is materially below 0.05**, so the result is not just visually better; it is statistically defensible.
* The confidence interval for uplift stays above zero, so the improvement is both significant and directionally reliable.
* Logistic regression shows the effect remains after controlling for device type, customer type, cart value, and time spent.
""")

add_code("""decision_summary = pd.DataFrame({
    'Metric': [
        'Control conversion rate',
        'Variant conversion rate',
        'Absolute uplift',
        'Relative uplift',
        'Z-statistic',
        'P-value',
        '95% CI for uplift'
    ],
    'Value': [
        f'{control_p:.2%}',
        f'{variant_p:.2%}',
        f'{uplift:.2%}',
        f'{relative_lift:.2%}',
        f'{z_stat:.4f}',
        f'{p_value:.6f}',
        f'[{ci_lower:.2%}, {ci_upper:.2%}]'
    ]
})

display(decision_summary)
""")

add_md("""**Interpretation:**
This table condenses the experiment into the exact facts needed for a go/no-go meeting. It makes the test choice, p-value, and effect size visible in one place.

---

## 10. Conclusion & Recommendations

### Summary of Key Findings:
1. **Clear Winner:** The "One-Click Checkout" (Variant) outperformed the traditional flow (Control) with an estimated relative uplift of roughly **24.5%** in this seeded run.
2. **Statistical Rigor:** The formal statistical test was a **two-proportion z-test** with the directional hypotheses `H0: p_new <= p_old` and `H1: p_new > p_old`. The p-value is clearly reported and falls well below 0.05.
3. **Mobile Friction:** While the variant helped all platforms, Mobile conversion remains structurally lower than Desktop. 
4. **Loyalty Pays:** Returning customers convert at extraordinarily higher rates, acting as the backbone of site revenue.

### Recommendations:
* **Launch Decision:** We reject `H0`, which means the evidence supports a statistically significant improvement in conversion. The recommendation is to launch the new checkout.
* **Next Step - Mobile Optimization:** Initiate a new UX research phase specifically targeting Mobile users. Even with the new checkout, Mobile OR sits at roughly **0.77** relative to Desktop. We need to investigate upstream friction (e.g., product discovery, page load speeds on 4G) for mobile users.
* **Next Step - Cart Value Segmentation:** Analyze if *extremely* high cart values (>$200) require a different flow (e.g., offering financing options via Affirm/Klarna instead of just one-click) to optimize AOV alongside conversion rate.

### Limitations:
* **Time-Series Effects (Novelty Effect):** This test simulated a 2-week period. It is possible the massive lift is partially driven by returning users being "surprised" by the faster flow (Novelty Effect). We should monitor the conversion rate over the next 6 weeks post-launch to establish the true long-term plateau.
* **Cannibalization Check:** We measured conversion events, but we must verify via financial databases that average order value (AOV) did not inadvertently drop (e.g., users accidentally checking out before adding secondary items).
""")


import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(project_root, "Conversion_Optimization_Analysis.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}, f, indent=1)

print(f"Notebook generated successfully at: {output_path}")
