import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


asthma_an = pd.read_csv("../data/asthma_disease_data_analysis.csv")
predictors = asthma_an.drop(columns="diagnosis")

X_train, X_test, y_train, y_test = train_test_split(predictors, asthma_an.diagnosis, test_size=0.3, random_state=42)

# Fit logistic regression
model = LogisticRegression(max_iter=1000)  # increase max_iter if needed
model.fit(X_train, y_train)

## Make predictions
# Predicted probabilities for class 1
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Predicted class labels
y_pred = model.predict(X_test)

## Evaluate performance
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# ROC AUC
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))

# Confusion matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
