# 🏦 End-to-End Credit Risk Prediction Pipeline

**Author:** [fatihh240](https://github.com/fatihh240)

## 🌐 Interactive Web Application (Live Demo)
The statistical model developed in this notebook has been successfully deployed as a full-stack data application. You can test the model's decision boundaries, simulate applicant profiles, and observe threshold dynamics interactively:

👉 **[Launch Credit Risk App - Streamlit Deployment](https://fatih-credit-risk-app.streamlit.app/)**

## 💾 Data Provenance
This pipeline is built upon the foundational **[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)**. 

---

## 📊 Project Architecture & Statistical Objective
This notebook demonstrates a complete machine learning pipeline for credit default risk assessment. Rather than treating the algorithm as a black box, this project focuses on robust feature engineering, understanding data variance, and optimizing decision thresholds for cost-sensitive evaluation.

**Methodology & Key Components:**
* **Mathematical Transformations:** Applying log(1+x) transforms to normalize heavily skewed financial distributions (e.g., income, loan amounts).
* **Strategic Data Binning:** Converting continuous variables into statistically significant ordinal groups to capture non-linear risk patterns and reduce noise.
* **Modeling:** Implementing a robust `LightGBM` classifier, optimized for high-dimensional tabular financial data.
* **Threshold Optimization:** Moving beyond the naive 0.5 probability threshold to establish business-aligned decision boundaries based on hypothetical loss-given-default (LGD) metrics.
