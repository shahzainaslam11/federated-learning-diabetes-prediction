# 🧠 Explainable Federated Learning with Differential Privacy for Type-2 Diabetes Readmission Prediction

> A research-grade, reproducible machine learning framework integrating **Federated Learning (FL)**, **Differential Privacy (DP)**, **Class Imbalance Handling**, and **Explainable AI (XAI)** for predicting 30-day hospital readmission in Type-2 Diabetes patients.

---

## 📄 Associated Publication

This repository implements the methodology described in the ACM-style research paper.

---

## 📌 Abstract

Hospital readmission within 30 days remains a critical challenge in healthcare systems, particularly for Type-2 Diabetes Mellitus (T2DM) patients. Traditional centralized machine learning approaches are constrained by **privacy risks, regulatory barriers, and data silos**.

This repository presents a **privacy-preserving federated learning framework** that:

- Enables collaborative training across multiple institutions **without data sharing**
- Incorporates **Differential Privacy (ε = 1.0, δ = 10⁻⁵)** to provide formal privacy guarantees
- Handles **class imbalance (~9:1)** using SMOTE and SMOTE-ENN
- Applies **Explainable AI (SHAP)** for interpretability

The framework achieves strong predictive performance while maintaining privacy and clinical interpretability.

---

## 🎯 Key Contributions

- 🔐 **Federated Learning Framework** simulating multi-hospital collaboration (5 clients)
- 🛡️ **Differential Privacy Integration** (gradient clipping + Gaussian noise)
- ⚖️ **Imbalance Handling** via SMOTE and SMOTE-ENN
- 🌳 **Ensemble Models**: Random Forest, XGBoost, LightGBM
- 📊 **Explainability via SHAP** for clinical insights
- 🔁 **Reproducible Pipeline** with config-driven experiments and fixed seeds

---

## 🧬 Methodology Overview

### 1. Data Pipeline

- Dataset: **UCI Diabetes 130-US Hospitals (101,766 records)**
- Preprocessing:
  - Remove identifiers and high-missing features
  - Median (numeric) & mode (categorical) imputation
  - Label encoding
  - Binary target: `<30` → 1, else 0

---

### 2. Federated Learning Setup

- **Clients:** 5 simulated hospitals
- **Algorithm:** FedProx (extension of FedAvg)
- **Communication Rounds:** 5–10
- **Local Training:** 3 epochs per round
- **Aggregation:** Weighted averaging
- **Prediction:** Ensemble voting across models

---

### 3. Differential Privacy Mechanism

- **Gradient Clipping:** L2 norm (C = 1.0)
- **Noise Injection:** Gaussian noise (σ ≈ 7.44)
- **Privacy Budget:** ε = 1.0, δ = 10⁻⁵

👉 Ensures formal privacy guarantees while maintaining utility

---

### 4. Class Imbalance Strategy

- SMOTE (synthetic minority oversampling)
- SMOTE-ENN (SMOTE + noise cleaning)

📌 Observations:
- SMOTE-ENN improves **F1-score & ROC-AUC**
- Slight trade-off in accuracy

---

### 5. Evaluation Protocol

- **Validation:** Stratified 10-Fold Cross Validation
- **Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

---

### 6. Explainability (XAI)

- SHAP used for:
  - Global feature importance
  - Local prediction explanations

📊 Top features:
- number_diagnoses  
- discharge_disposition_id  
- time_in_hospital  
- num_inpatient  
- age  

---

## 📁 Repository Structure

```bash
.
├── configs/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
├── scripts/
├── experiments/
├── tests/
├── docs/
├── README.md
└── requirements.txt
````

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/diabetes-fl-dp.git
cd diabetes-fl-dp

pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Train Model

```bash
python scripts/train.py
```

### 🔹 Run Full Experiment (All Models + Sampling)

```bash
python scripts/run_experiment.py
```

### 🔹 Evaluate Model

```bash
python scripts/evaluate.py
```

---

## 📊 Results Summary

| Model         | Accuracy | F1 Score | ROC-AUC |
| ------------- | -------- | -------- | ------- |
| Random Forest | ~0.84    | ~0.87    | ~0.91   |
| XGBoost       | ~0.92    | ~0.93    | ~0.96   |
| LightGBM      | ~0.93    | ~0.93    | ~0.96   |

📌 Observations:

* XGBoost & LightGBM outperform RF
* SMOTE-ENN improves minority detection
* DP introduces minimal performance drop

---

## 🔁 Reproducibility

This repository ensures strict reproducibility:

* Fixed random seed (`seed = 42`)
* Config-driven experiments
* Deterministic preprocessing
* Structured logging
* Modular pipeline

---

## 🧪 Testing

```bash
pytest tests/
```

Includes:

* Data pipeline validation
* Model forward pass checks

---

## 💻 Environment

* Python: 3.9+
* scikit-learn: 1.3+
* XGBoost: 2.0+
* LightGBM: 4.1+
* SHAP: 0.43+

**Hardware:**

* CPU compatible
* GPU optional (for boosting models)

---

## ⚠️ Limitations (Critical for Reviewers)

* No real-world federated deployment (simulation only)
* No external validation dataset
* Potential bias from resampling methods
* Communication efficiency not optimized

---

## 🔬 Future Work

* Real multi-hospital deployment
* Adaptive privacy budgets (dynamic ε)
* Secure aggregation (cryptographic)
* External validation datasets
* Communication-efficient FL (compression, sparsification)
* Deep learning integration (tabular transformers)

---

## 📚 Citation

```bibtex
@article{aslam2026fed_dp_diabetes,
  title={Explainable Federated Learning for Privacy Preserving Type-2 Diabetes Prediction},
  author={Aslam, Shahzain and Shahid, Zohaib and Ahmed, Irfan},
  journal={ACM Conference Proceedings},
  year={2026}
}
```

---

## 🙏 Acknowledgements

* UCI Machine Learning Repository
* Healthcare ML research community
* Federated learning and privacy-preserving AI researchers

---

## 📬 Contact
For research collaboration or queries:

📧 [Shahzainaslam28@gmail.com](mailto:Shahzainaslam28@gmail.com)
