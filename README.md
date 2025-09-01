<div align="center">

# ❤️ Heart Disease Risk Prediction

**AI-powered heart disease risk prediction & exploratory analytics platform.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#) [![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
<sub>Educational / research use only – *not* a medical diagnostic tool.</sub>

</div>

---

## 🩺 Overview
This repository contains an end‑to‑end machine learning workflow to estimate the probability of heart disease based on structured clinical indicators. It spans **data acquisition, cleaning, exploratory data analysis (EDA), feature engineering & selection, dimensionality reduction, supervised & unsupervised modeling, hyperparameter optimization, decision threshold tuning, model packaging, and deployment via a Streamlit web application** with interactive risk & confidence visualization.

> **Disclaimer:** The model and app are for **educational and research purposes only** and must **not** be used for real clinical decision-making.

## Team
**Yousef Salah Nage**  
AI & Machine Learning Enthusiast  
LinkedIn: https://www.linkedin.com/in/yousef-salah-nage-a3583636b  
GitHub: https://github.com/YousefSalah123  
**Abdelrahman Mohsen**
AI & Machine Learning Enthusiast  
LinkedIn: https://www.linkedin.com/in/abdelrahman-mohsen5600?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app 
GitHub: https://github.com/3bood5600


## 📊 Dataset
Based on a heart disease clinical indicators dataset (UCI-style schema). Each record includes demographics (age, sex), vitals (resting blood pressure, cholesterol), stress test results (exercise-induced angina, ST depression, peak heart rate), and categorical cardiology assessments (chest pain type, slope, thalassemia markers).  

### Key Preprocessing & Engineering Steps
- Missing value handling & type normalization
- Outlier awareness (IQR / distribution review)
- Feature engineering: `chol_per_age`, `heart_rate_reserve`, and domain-driven categorical one-hot encodings
- Standardization of numeric features (`StandardScaler` inside pipeline)
- One-hot encoding of categorical features with unknown-safe handling
- Optional dimensionality reduction exploration via **PCA** (variance retention analysis)
- Multiple feature selection strategies: Random Forest & XGBoost importance, RFE, Chi-Square tests – consolidated into a stable subset (`selected_features.csv`)

## 🧪 Project Workflow
| Stage | Notebook(s) | Highlights |
|-------|-------------|-----------|
| Data Cleaning & EDA | `01_data_preprocessing`, `Final_EDA_Notebook` | Schema audit, distributions, correlations, class balance |
| PCA Analysis | `02_pca_analysis` | Variance explained, component interpretation (exploratory) |
| Feature Selection | `03_feature_selection` | Importance fusion (RF / XGBoost), RFE, Chi² scoring, final subset export |
| Supervised Modeling | `04_supervised_learning`, `05_supervised_learning_comparison`, `Final_Modelling` | Logistic Regression, Decision Tree, Random Forest, SVM benchmarking |
| Unsupervised Learning | `05_unsupervised_learning`, `06_unsupervised_learning` | K-Means, Hierarchical clustering – structure & grouping insight |
| Hyperparameter Tuning | `06_hyperparameter_tuning`, `07_hyperparameter_tuning` | GridSearchCV / RandomizedSearchCV, CV diagnostics |
| Model Refinement & Ensemble | comparative notebooks | Threshold analysis, probability calibration review, soft voting & stacking trials |
| Final Export | `08_model_export_and_deployment` | Persist best pipeline & metadata (`best_model.pkl`, `best_model_report.json`) |
| Streamlit App | `ui/app.py` | Interactive prediction, risk gradient bar, confidence gauge, insights pages |

## 🏆 Final Model
**Model:** Threshold‑Tuned SVC (`rbf` kernel)  
**Best Probability Threshold:** `0.617` (optimizes balance of recall & precision)  
**Core Preprocessing:** Standardized numerics + one-hot categorical features within pipeline.  

| Metric | Value |
|--------|-------|
| Accuracy | 0.875 |
| Precision | 0.8911 |
| Recall | 0.8824 |
| F1 Score | 0.8867 |
| ROC AUC | 0.9063 |

> Stored in `results/best_model_report.json` along with feature lists & preprocessing metadata.

## 🖥️ Streamlit Web Application
The app provides:
1. **Prediction Form** – Structured user input with validation
2. **Risk Score Visualization** – Horizontal gradient bar (green→orange→red) + threshold marker
3. **Confidence Gauge** – Model confidence (% max(p, 1-p))
4. **Risk Category & Threshold Context** – Clear label (Low / Moderate / High)
5. **Feature Vector Preview & Export** – Downloadable CSV of the inference row
6. **Models & Results Page** – Summary of comparative performance
7. **Data & Insights Page** – Exploratory charts (class balance, distributions, correlations)

Run it locally (see Installation).

## 📂 Project Structure
```
├── data/
│   └── selected_features.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   ├── 01_data_preprocessing (1).ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_supervised_learning_comparison.ipynb
│   ├── 05_unsupervised_learning.ipynb / 06_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb / 07_hyperparameter_tuning.ipynb
│   ├── 08_model_export_and_deployment.ipynb
│   └── Final_* (reference / consolidated)
├── results/
│   ├── best_model_report.json
│   ├── best_models_summary.json
│   └── tuning_cv_results.csv
├── ui/
│   └── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Usage
> Requires Python 3.10+ (tested) and pip.

```bash
git clone <repo_url>
cd Heart_Disease_Project - Fiinal
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run ui/app.py
```

### Programmatic Inference Example
```python
import joblib, pandas as pd, json
model = joblib.load('models/best_model.pkl')
with open('results/best_model_report.json') as f:
    report = json.load(f)
features = report['preprocessing']['numeric_features'] + report['preprocessing']['categorical_features']
# df_new must contain ALL engineered & encoded columns
proba = model.predict_proba(df_new[features])[:,1]
pred = (proba >= report['best_threshold']).astype(int)
```

## 📈 Results & Visualizations
- **ROC AUC ~0.91** indicates strong discriminative ability.
- Calibrated threshold (0.617) improves balance vs naive 0.50 rule.
- Engineered features (`chol_per_age`, `heart_rate_reserve`) contributed to lift in recall.
- Feature importance & correlation analyses guide interpretability; clustering offered exploratory subgroup patterns (non-deployment).

Potential example visuals (not embedded here):
- ROC Curve & Precision-Recall Curve
- Risk Probability Gradient Bar
- Feature Importance Bars
- Class Distribution & Correlation Heatmap

## 🔄 Reproducibility & MLOps Principles
- Deterministic seeds (`random_state=42`)
- Immutable saved artifacts (`best_model.pkl`, reports, feature subsets)
- Separation of **data prep / selection / modeling / evaluation / deployment** stages
- Clear threshold documentation for decision layer
- Modular transformation inside pipeline to minimize leakage

## 🚀 Future Work
- Add SHAP / permutation importance for model interpretability
- Expand dataset (multi-source aggregation & temporal variables)
- Try gradient boosting variants (LightGBM, CatBoost) & calibrated probabilities
- Deploy via container + CI/CD (GitHub Actions) & cloud hosting
- Add monitoring: drift detection, periodic re-training triggers
- Privacy & security hardening (PII audits, model card)

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer
This project does **not** provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for clinical decisions.

---

<div align="center"><sub>Crafted with data science rigor for learning & experimentation.</sub></div>
