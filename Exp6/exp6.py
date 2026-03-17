import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("loan_data.csv")

# Features & Target
X = df[['person_income','loan_amnt','credit_score']]
y = df['loan_status']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Set MLflow Experiment
# -------------------------------
mlflow.set_experiment("Loan_Model_Experiment")

# -------------------------------
# Function to train + log
# -------------------------------
def train_model(C_value):

    with mlflow.start_run():

        # Log hyperparameter
        mlflow.log_param("C", C_value)

        # Model (hyperparameter changed here)
        model = LogisticRegression(C=C_value, max_iter=200)

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Save locally
        pickle.dump(model, open("model.pkl", "wb"))

        # Log model in MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"Run done: C={C_value}, Accuracy={accuracy}")


# -------------------------------
# Run experiments (IMPORTANT)
# -------------------------------
train_model(1.0)    # Run 1
train_model(0.5)    # Run 2 (changed hyperparameter)

print("All runs complete. Run 'mlflow ui'")