import os
import json
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Konfigurasi DagsHub

dagshub.init(
    repo_owner="rrralsaputra",
    repo_name="MembangunSistemMachineLearning",
    mlflow=True
)

mlflow.set_experiment("Heart Disease Classification Tuning")

# Load Dataset

dataset_path = dataset_path = "D:\CodingCamp\PIJAK\ML\Eksperimen_SML_Muhammad Geralldo\preprocessing\heart_preprocessing.csv"
df = pd.read_csv(dataset_path)

target_column = "target"

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tuning Manual

n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        run_name = f"RF_n{n_estimators}_depth{max_depth}"

        with mlflow.start_run(run_name=run_name):

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Manual logging parameter
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)

            # Manual logging metric
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Logging model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            # Artefak tambahan
            artifact_dir = f"artifacts/{run_name}"
            os.makedirs(artifact_dir, exist_ok=True)

            # Artefak 1: Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Confusion Matrix {run_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            cm_path = f"{artifact_dir}/confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()

            mlflow.log_artifact(cm_path)

            # Artefak 2: Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)

            report_path = f"{artifact_dir}/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

            mlflow.log_artifact(report_path)

            print(f"Run selesai: {run_name}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1}")