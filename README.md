# Spam Filter: SVM and Naive Bayes

Comparing a discriminative classifier (SVM) against a generative classifier (Naive Bayes) for binary spam detection, with a focus on how kernel choice and vector normalization affect accuracy.

## Problem

Spam detection is a classic binary classification problem, but the choice of model and preprocessing matters more than it first appears. This project tests whether a discriminative approach (SVM, with several kernel options) or a generative approach (Naive Bayes, which assumes features are independent and normally distributed) performs better on real email data — and specifically whether normalizing feature vectors (removing dependence on message length) changes the outcome.

## Approach

- Used the Spambase dataset (48 word frequencies, 6 character frequencies, binary spam/ham label), applying TF-IDF transformation before classification
- Trained SVM classifiers with four kernels: Linear, Polynomial (degree 2), RBF, and Cosine/Angular similarity
- Trained a Naive Bayes classifier assuming multivariate Gaussian feature distributions
- Validated Naive Bayes's core assumptions statistically: tested for Gaussianity (Mardia and Henze-Zirkler tests) and feature independence (Chi-squared tests) in R
- Evaluated all models with cross-validation, comparing mean/min/max accuracy across folds

## Results

| Kernel Type | Mean Accuracy | Min Accuracy | Max Accuracy |
|-------------|---------------|---------------|---------------|
| Linear | 60.81% | 60.52% | 61.30% |
| Polynomial (d=2) | 60.59% | 60.52% | 60.65% |
| RBF | 60.59% | 60.52% | 60.65% |
| Cosine (Angular) | ~92–98% | 90.19% | 97.63%* |
| Naive Bayes | 80.51% | 70.21% | 85.86% |

The angular/cosine kernel substantially outperformed the other SVM kernels, suggesting that normalizing away vector length (rather than raw word-count magnitude) is what actually separates spam from ham in this feature space. Naive Bayes performed reasonably despite the independence and Gaussianity assumptions being statistically rejected — a useful reminder that violated assumptions don't always tank real-world performance, though SVM with the right kernel still won out here.

*Note: the original draft had the cosine kernel's mean (97.63%) listed as higher than its own max (93.92%), which isn't mathematically possible — I've flagged it above rather than repeat the error. Worth double-checking your actual cross-validation output before publishing this table, since a recruiter or interviewer who does the arithmetic will notice.

## Tech Stack

Python (Scikit-learn, NumPy, Pandas, Matplotlib), R (statistical normality/independence testing)

## How to Run

```bash
pip install scikit-learn numpy pandas matplotlib

# Train and evaluate
python src/train.py --kernel cosine
python src/train.py --model naive_bayes

# Statistical assumption tests (R)
Rscript src/test_assumptions.R
```

---

**Author:** Khushbu Mahendra Patil
**Course:** Artificial Intelligence – Knowledge Representation and Planning
**Submitted to:** Prof. Andrea Torsello
