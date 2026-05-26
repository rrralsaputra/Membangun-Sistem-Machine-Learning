import os
import json
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

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

# Load dataset
dataset_path = "heart_preprocessing.csv"
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

# Training model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Folder output
os.makedirs("artifacts", exist_ok=True)
if os.path.exists("model"):
    shutil.rmtree("model")

# Simpan classification report
report = classification_report(y_test, y_pred, output_dict=True)

with open("artifacts/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

# Simpan confusion matrix sederhana
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png")
plt.close()

# Simpan model dalam format MLflow
mlflow.sklearn.save_model(
    sk_model=model,
    path="model",
    input_example=X_test.iloc[:5]
)

# Simpan metrik ke file json agar bisa jadi artifact GitHub
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training selesai")
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")
