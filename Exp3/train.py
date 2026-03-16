import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
df = pd.read_csv("loan_data.csv")

# Convert categorical columns to numbers
df = pd.get_dummies(df)

# Separate features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
specificity = TN / (TN + FP)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Specificity:", specificity)
print("F1 Score:", f1)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")