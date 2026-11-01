import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load employee data
df = pd.read_csv("employee_data.csv")

# Remove unnecessary columns
df = df.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

# Convert target column
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Convert text columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Employee Attrition Analysis")
print("---------------------------")

print("Accuracy:", accuracy)

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Save results
results = X_test.copy()

results["Actual_Attrition"] = y_test
results["Predicted_Attrition"] = predictions

results.to_csv(
    "employee_report.csv",
    index=False
)

print("Employee report saved successfully!")