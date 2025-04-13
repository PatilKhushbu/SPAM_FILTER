# 📧 Spam Filter with Support Vector Machine and Naive Bayes

This project implements and compares discriminative and generative classifiers to build a **spam filter**, using the [Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase). The classifiers are:

- 🧠 **Support Vector Machine (SVM)** with various kernels (Linear, Polynomial, RBF, and Angular)
- 📊 **Naive Bayes** assuming multivariate Gaussian distributions

## 🗂️ Project Structure

```
📁 spam-filter-classifier/
├── data/              # Preprocessed dataset (TF-IDF transformed)
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks for experimentation
├── src/               # Source code: training, evaluation, utils
├── results/           # Cross-validation results and plots
└── README.md
```

## 📌 Objective

> Build a binary spam filter using machine learning, compare the effectiveness of discriminative and generative models, and explore the impact of vector normalization (angular information).

---

## 🧪 Dataset

The **Spambase** dataset represents email messages with:
- 48 word frequencies
- 6 character frequencies
- 1 binary label (spam = 1, ham = 0)

After removing irrelevant features, we apply **TF-IDF** transformation before classification.

---

## 🔍 Classification Models

### ✅ Support Vector Machine (SVM)

- **Kernels Used**:
  - Linear
  - Polynomial (Degree = 2)
  - Radial Basis Function (RBF)
  - Cosine similarity (Angular kernel)

- **Highlights**:
  - Performance improves significantly with vector normalization
  - Achieves up to **97.6% accuracy** with cosine kernel

### 📈 Naive Bayes

- Assumes **conditional independence** and **multivariate normality**
- Fast training time
- Less accurate than SVM (~80% mean accuracy)

- **Assumption Validation**:
  - ❌ Mardia and Henze-Zirkler tests reject Gaussianity
  - ❌ Chi-squared tests reject feature independence

---

## ⚙️ Tools & Libraries

- Python 3.x
- Scikit-learn (`SVM`, `NaiveBayes`, `TF-IDF`, `Cross-validation`)
- NumPy, Pandas, Matplotlib
- R (for statistical normality tests)

---

## 📊 Results

| Kernel Type           | Mean Accuracy | Min Accuracy | Max Accuracy |
|-----------------------|---------------|---------------|---------------|
| Linear                | 60.81%        | 60.52%        | 61.30%        |
| Polynomial (d=2)      | 60.59%        | 60.52%        | 60.65%        |
| RBF                   | 60.59%        | 60.52%        | 60.65%        |
| **Cosine (Angular)**  | **97.63%**    | 90.19%        | 93.92%        |
| **Naive Bayes**       | **80.51%**    | 70.21%        | 85.86%        |

> 🔍 Angular kernels outperform traditional ones by removing vector length dependency.

---

## 🧠 Key Insights

- SVM with angular kernels (cosine similarity) is most effective for text-based features.
- Naive Bayes is computationally efficient but suffers if data assumptions are violated.
- Normalizing vectors before classification improves robustness and accuracy.

---

## 📌 Authors

- **Khushbu Mahendra Patil** — Matric. No: 882697  
- **Khalid Vafa** — Matric. No: 882677  
- 📘 Course: Artificial Intelligence – Knowledge Representation and Planning  
- 👨‍🏫 Submitted to: Prof. Andrea Torsello

---
