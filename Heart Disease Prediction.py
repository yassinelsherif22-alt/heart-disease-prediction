import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("heart.csv")

print("\n=== Dataset Information ===")
print(df.info())

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Statistical Summary ===")
print(df.describe())

print("\n=== Missing Values Before Filling ===")
print(df.isnull().sum())

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()


X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', C=10.0, random_state=0),
    "Support Vector Machine": SVC(C=6, kernel='rbf', gamma=0.1),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights='distance')
}

results = {}

for name, model in models.items():
    if name in ["Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': cr
    }

    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

print("\n=== Model Comparison ===")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.2f}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.2f}")

best_result = results[best_model_name]
plt.figure(figsize=(6, 4))
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix for {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{best_model_name.lower().replace(' ', '_')}_confusion_matrix.png")
plt.close()

new_patient = pd.DataFrame([{
    'age': 20,
    'sex': 1,
    'cp': 0,
    'trestbps': 110,
    'chol': 250,
    'fbs': 0,
    'restecg': 1,
    'thalachh': 150,
    'exang': 0,
    'oldpeak': 1.2,
    'slope': 2,
    'ca': 0,
    'thal': 3
}])

best_model = models[best_model_name]
if best_model_name in ["Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression"]:
    new_patient_scaled = scaler.transform(new_patient)
    prediction = best_model.predict(new_patient_scaled)
else:
    prediction = best_model.predict(new_patient)

print("\n=== New Patient Prediction ===")
if prediction[0] == 1:
    print("The patient is likely to have heart disease.")
else:
    print("The patient is unlikely to have heart disease.")

















