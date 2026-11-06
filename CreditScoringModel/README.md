# Credit Scoring Model

A machine learning project for credit risk classification using financial and credit-history features. The project implements and compares multiple classification algorithms to demonstrate how machine learning can support credit risk assessment.

## Project Overview

Credit risk assessment helps financial institutions evaluate the potential risk associated with lending money. This project demonstrates a supervised machine learning approach to classify applicants into two illustrative categories: Good Credit and Bad Credit.

Three machine learning algorithms are implemented and evaluated using standard classification metrics.

**Note:** The current implementation uses synthetically generated data for educational purposes. Its performance metrics do not represent real-world credit risk accuracy.

## Objectives

- Build a binary credit risk classification model.
- Prepare financial data for machine learning.
- Train and compare multiple classification algorithms.
- Evaluate performance using standard classification metrics.
- Visualize model results using charts and a confusion matrix.
- Predict the credit risk category of a sample applicant.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Algorithms

### Logistic Regression
A classification algorithm used to estimate the probability of an applicant belonging to a credit risk category.

### Decision Tree
A tree-based model that learns decision rules from input features.

### Random Forest
An ensemble learning algorithm that combines multiple decision trees to generate predictions.

## Dataset Description

The current version generates a synthetic dataset containing 1,000 sample records.

### Input Features

| Feature | Description |
|---|---|
| Income | Applicant's income |
| Debt | Existing debt amount |
| Credit History | Length of credit history |
| Missed Payments | Number of missed payments |
| Employment Years | Years of employment |
| Loan Amount | Requested loan amount |

### Target Variable

- `0` — Good Credit
- `1` — Bad Credit

The target labels are generated using an artificial scoring rule and are not actual borrower outcomes.

## Model Evaluation

The models are evaluated using:

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** Proportion of predicted bad-credit cases that are labelled bad credit.
- **Recall:** Proportion of actual bad-credit cases correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model separates the two classes across different thresholds.

The project also generates a model comparison chart and a confusion matrix.

## Project Structure

```text
CreditScoringModel/
├── credit_scoring.py
├── requirements.txt
├── README.md
├── .gitignore
├── model_comparison.png
└── confusion_matrix.png
```

The image files are generated when the Python program runs successfully.

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- Visual Studio Code or another Python editor

### 1. Install Dependencies

Open the terminal in the project directory and execute:

```bash
python -m pip install -r requirements.txt
```

### 2. Run the Project

```bash
python credit_scoring.py
```

The program displays the model evaluation results, classification report, and sample applicant prediction. It also generates and saves visualization images.

## Expected Outputs

- Dataset preview
- Model performance comparison table
- Accuracy, precision, recall, F1-score, and ROC-AUC
- Classification report
- Model comparison graph
- Confusion matrix
- Sample applicant credit risk prediction

## Limitations and Future Enhancements

- Integrate a real, appropriately licensed credit dataset.
- Improve feature engineering and data preprocessing.
- Apply cross-validation and hyperparameter tuning.
- Evaluate class imbalance and prediction errors.
- Investigate model fairness, bias, and explainability.
- Validate predicted probabilities before interpreting them as actual risk estimates.

## Learning Outcomes

This project demonstrates practical concepts in supervised learning, binary classification, feature scaling, model evaluation, data visualization, and prediction using Scikit-learn.

## Disclaimer

This project is intended for educational purposes only. It is not a validated financial assessment tool and should not be used as the sole basis for real lending decisions.

---

**Developed using Python and Scikit-learn.**