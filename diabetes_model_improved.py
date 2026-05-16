import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import pickle

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Handle zero values as missing
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("🔧 Best Parameters:", grid.best_params_)

# Evaluate the tuned model
y_pred = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, probs))

# Cross-validation accuracy
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("📊 Cross-validation Accuracy Scores:", cv_scores)
print("📈 Mean CV Accuracy:", cv_scores.mean())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Diabetic", "Diabetic"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save model & scaler
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Final model and scaler saved successfully!")
