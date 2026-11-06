
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# 1. Generate demonstration dataset
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "income": np.random.randint(15000, 150001, n),
    "debt": np.random.randint(0, 100001, n),
    "credit_history": np.random.randint(0, 21, n),
    "missed_payments": np.random.randint(0, 9, n),
    "employment_years": np.random.randint(0, 31, n),
    "loan_amount": np.random.randint(5000, 200001, n)
})

# 2. Create sample target labels
risk_score = (
    1.5 * data["debt"] / 100000
    + 0.8 * data["loan_amount"] / 200000
    + 0.7 * data["missed_payments"]
    - 1.0 * data["income"] / 150000
    - 0.5 * data["credit_history"] / 20
    - 0.3 * data["employment_years"] / 30
    + np.random.normal(0, 0.8, n)
)

data["credit_risk"] = (risk_score > 2.5).astype(int)

print("CREDIT SCORING MODEL")
print("=" * 55)
print("\nFirst five records:")
print(data.head())
print("\nDataset shape:", data.shape)
print("\nTarget distribution:")
print(data["credit_risk"].value_counts())

# 3. Separate features and target
X = data.drop("credit_risk", axis=1)
y = data["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Define machine learning models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    )
}

results = []
trained_models = {}

# 5. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    trained_models[name] = model

# 6. Display results
results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE")
print(results_df.round(3).to_string(index=False))

best_name = results_df.loc[
    results_df["F1-Score"].idxmax(), "Model"
]
best_model = trained_models[best_name]

print("\nBest model by F1-score:", best_name)

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    target_names=["Good Credit", "Bad Credit"],
    zero_division=0
))

# 7. Plot model comparison and save image
results_df.set_index("Model").plot(
    kind="bar", figsize=(11, 6), ylim=(0, 1)
)
plt.title("Credit Scoring Model Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

# 8. Plot and save confusion matrix
predictions = best_model.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Good Credit", "Bad Credit"],
    yticklabels=["Good Credit", "Bad Credit"]
)
plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 9. Predict a new sample applicant
applicant = pd.DataFrame([{
    "income": 60000,
    "debt": 20000,
    "credit_history": 8,
    "missed_payments": 1,
    "employment_years": 5,
    "loan_amount": 30000
}])

prediction = best_model.predict(applicant)[0]
probability = best_model.predict_proba(applicant)[0, 1]

print("\nNEW APPLICANT PREDICTION")
print(applicant.to_string(index=False))
print("Predicted category:",
      "Bad Credit" if prediction == 1 else "Good Credit")
print("Estimated bad-credit probability:",
      round(probability * 100, 2), "%")

print("\nProject completed successfully!")
print("Note: Synthetic data is used for demonstration only.")