<<<<<<< HEAD
# E-Commerce Conversion Optimization: Causal A/B Testing Framework
=======
# Conversion Optimization A/B Test: Should We Launch the New Checkout?
>>>>>>> f8f6060 (Updated.)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Analysis](https://img.shields.io/badge/Project-A%2FB%20Testing-orange.svg)]()
[![Decision](https://img.shields.io/badge/Focus-Decision%20Making-success.svg)]()

<<<<<<< HEAD
A professional, end-to-end data analysis project demonstrating the lifecycle of an E-Commerce A/B experiment using statistical and causal inference techniques.

---

## Abstract

Businesses frequently run A/B tests to improve product features such as checkout flows. However, naive analysis, such as comparing average conversion rates, can lead to misleading conclusions due to confounding variables and Simpson’s Paradox.

Using simulated data designed with controlled confounders (device type and customer loyalty), this project presents a structured approach to experimentation:

- Frequentist hypothesis testing (Z-Test)  
- Logistic regression for causal inference  
- Bayesian A/B testing for probability-based decision making  
- Business impact simulation for revenue estimation  

---

## Problem Statement

An e-commerce platform is experiencing high drop-off during the checkout process. A new “one-click checkout” feature is proposed to reduce friction and improve conversion rates.

The objective is to evaluate whether the new checkout flow significantly improves conversion while controlling for user behavior, device type, and customer characteristics.

---

## Repository Structure

```text
conversion-optimization-ab-testing/
│
├── README.md
├── requirements.txt
├── Conversion_Optimization_Analysis.ipynb
│
├── src/
│   ├── frequentist_ab.py
│   ├── power_analysis.py
│
├── scripts/
│   └── create_notebook.py
│
├── assets/
│   ├── conversion_rate.png
│   ├── odds_ratio_plot.png
│   └── device_conversion.png
│
└── .gitignore

```
## Tech Stack

- numpy, pandas (data manipulation)  
- matplotlib, seaborn (data visualization)  
- scipy (statistical testing)  
- statsmodels (logistic regression)  
- scikit-learn (preprocessing and scaling)  

---

## How to Run the Analysis

### Clone the repository
```bash
git clone https://github.com/the-irritater/conversion-optimization-ab-testing.git
cd conversion-optimization-ab-testing
```
---

**Explore the Notebook:** Open Conversion_Optimization_Analysis.ipynb using Jupyter Notebook, JupyterLab, or Visual Studio Code. 
Be sure to click **"Run All"** to execute the cells and view the saved visualizations and outputs. 
*(Note: The notebook can be cleanly re-generated from scratch locally at any time by running python create_notebook.py)* 

## Key Findings 
**Treatment Effect:** The "One-Click Checkout" variant successfully created an 18% relative uplift in conversion rate. *
**Causal Significance:** Even holding user demographics entirely constant, the variant reliably improves conversion odds by ~25% (O.R = ~1.25). * 
**Bayesian Outcome:** There is a nearly 100% probability that the variant outperforms the control, with almost zero expected loss. * 
**Business Value:** Projected $2.4M in annualized incremental revenue upon 100% rollout.*

---
## 📊 Key Insights

### Conversion Lift
![Conversion](assets/revenue_impact.png)

### Causal Impact (Logistic Regression)
![Odds Ratio](assets/odds_ratio_plot.png)

### Segment Analysis (Device Type)
![Device](assets/conversion_rate.png)

---

Conclusion

The analysis demonstrates that the proposed one-click checkout flow has a statistically significant and practically meaningful impact on conversion rates.

By combining traditional hypothesis testing with causal inference and Bayesian methods, this project highlights how data-driven experimentation can support confident business decision-making.

---

## Author

Sanman Kadam
MSc Statistics | Data Analyst | Data Science Enthusiast

GitHub: https://github.com/the-irritater

---

*Created as a demonstration of advanced statistical analysis and business intelligence methodologies.*
=======
This project is written like a real product decision, not a theory exercise.

The business question is simple:

Should an e-commerce team launch a new one-click checkout flow, or keep the old experience live?

The analysis walks from that question to a final recommendation using statistical evidence, effect-size interpretation, and business impact framing.

## The Story

The checkout team believes the new experience reduces friction and should improve conversion. That belief is not enough to justify a rollout. Launching too early could hurt revenue, create false confidence, and waste engineering effort.

So the decision flow in this project is:

1. Measure conversion for old vs new checkout.
2. Run a formal hypothesis test.
3. Show the p-value clearly.
4. Visualize the confidence interval around the uplift.
5. Control for customer and device differences with logistic regression.
6. Translate the result into a business recommendation.

## Decision Framework

### 1. Hypothesis Section

- `H0: p_new <= p_old`
- `H1: p_new > p_old`

This is a directional launch decision. We only want to ship if the new checkout is better.

### 2. Statistical Test

- Primary test: `Two-proportion z-test`
- Significance level: `alpha = 0.05`

The notebook reports:

- Control conversion rate
- Variant conversion rate
- Absolute uplift
- Relative uplift
- Z-statistic
- P-value
- 95% confidence interval for uplift

## Advanced Layer Added

This project includes an advanced layer beyond the basic hypothesis test:

- `Confidence interval visualization`

That matters because a p-value tells us whether the result is statistically significant, while the confidence interval shows the plausible range of business impact. It helps answer a better question:

Is the uplift only detectable, or is it large enough to matter?

## Business Conclusion Style

The analysis ends with a business-facing decision, not just a statistical statement.

Example decision language used in the notebook:

`We fail to reject H0 -> no significant improvement -> do not launch`

In the current simulated experiment, the result supports the opposite direction:

`We reject H0 -> statistically significant improvement -> launch is supported`

## Verified Output Snapshot

These are the current seeded results from the notebook:

- Control conversion rate: `19.42%`
- Variant conversion rate: `24.19%`
- Absolute uplift: `4.76 percentage points`
- Relative uplift: `24.52%`
- One-sided p-value: `3.96e-09`
- 95% confidence interval for uplift: `[3.15 pp, 6.38 pp]`
- Variant odds ratio: `1.33`
- Returning-customer odds ratio: `1.67`
- Mobile odds ratio: `0.77`
- Median AOV: `$33.09`
- Projected incremental annual revenue: `$1.89M`

## What Makes This Resume-Worthy

- It uses the correct A/B testing hypothesis structure for a product launch decision.
- It applies a named statistical test instead of relying on visual differences alone.
- It surfaces the p-value clearly instead of hiding it inside a long explanation.
- It adds an advanced interpretation layer with confidence intervals.
- It includes logistic regression to show the result still holds after controlling for customer mix and device mix.
- It ends with an executive-style recommendation tied to action.

## Repository Structure

```text
conversion-optimization-ab-testing/
|-- README.md
|-- requirements.txt
|-- Conversion_Optimization_Analysis.ipynb
|-- scripts/
|   |-- create_notebook.py
|-- src/
|   |-- frequentist_ab.py
|   |-- power_analysis.py
|-- assets/
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Regenerate the notebook if needed:

```bash
python scripts/create_notebook.py
```

3. Open and run:

```bash
Conversion_Optimization_Analysis.ipynb
```

## GitHub Upload Checklist

- README explains the business question and decision flow.
- Notebook includes the statistical test, p-value, confidence interval, and business conclusion.
- The committed notebook should be saved with outputs so GitHub renders the charts and tables.

## Core Takeaway

This project shows that strong portfolio work is not about adding more theory. It is about making the decision path obvious:

question -> test -> p-value -> uncertainty range -> business recommendation
>>>>>>> f8f6060 (Updated.)
