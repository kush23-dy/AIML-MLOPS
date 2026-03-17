import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("loan_data.csv")

# Features & Target
X = df[['person_income', 'loan_amnt', 'credit_score']]
y = df['loan_status']

# Convert target if not numeric
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MLflow Experiment
# -------------------------------
mlflow.set_experiment("Loan_Model_Experiment")

# -------------------------------
# Training Function
# -------------------------------
def train_model(C_value):

    with mlflow.start_run():

        # Log hyperparameter
        mlflow.log_param("C", C_value)

        # Model
        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # -------------------------------
        # Metrics
        # -------------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # -------------------------------
        # ROC Curve
        # -------------------------------
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # -------------------------------
        # Save Model
        # -------------------------------
        pickle.dump(model, open("model.pkl", "wb"))

        # Updated MLflow logging (no deprecation warning)
        mlflow.sklearn.log_model(model, name="model")

        print(f"Run done: C={C_value}, Accuracy={accuracy}")


# -------------------------------
# Run Experiments
# -------------------------------
train_model(1.0)   # Run 1
train_model(0.5)   # Run 2

print("All runs complete. Run 'mlflow ui'")