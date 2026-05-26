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

mlflow.set_experiment("Heart Disease Classification")

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

# Training Model

with mlflow.start_run(run_name="RandomForest_Baseline"):

    n_estimators = 100
    max_depth = 10
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Manual Logging Parameter

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("dataset", dataset_path)

    # Manual Logging Metric

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Logging Model

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )


    # Artefak Tambahan

    os.makedirs("artifacts", exist_ok=True)

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("artifacts/confusion_matrix.png")

    # Artefak 2: Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)

    with open("artifacts/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    mlflow.log_artifact("artifacts/classification_report.json")

    # Artefak 3: Feature Importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance.to_csv("artifacts/feature_importance.csv", index=False)
    mlflow.log_artifact("artifacts/feature_importance.csv")

    print("Training selesai.")
    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1 Score : {f1}")