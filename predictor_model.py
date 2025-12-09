# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# load data set
heart_data = pd.read_csv("framingham.csv")
print("Number of instances: " + str(len(heart_data)))

# check for duplicate values - none
print("Duplicate values: " + str(sum(heart_data.duplicated())))

# check for instances with null values
print("Null values by category: ")
print(heart_data.isnull().sum())

# drop instances with null values in specific columns
heart_data.dropna(subset=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate','glucose'], inplace=True)
print("Number of instances after dropping null: " + str(len(heart_data)))

# create test-train-split, accounting for imbalanced dataset using stratify=y (maybe look into SMOTE?)
y = heart_data[['TenYearCHD']]
X = heart_data.drop(['TenYearCHD'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.2, random_state=42, stratify=y)

# create initial simple logistic model (look into more hyperparameter tuning?)
CHD_predictor = LogisticRegression(max_iter=10000, penalty='l2')

# simple normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit model
CHD_predictor.fit(X_train_scaled, y_train)

# initial scoring:
print("\nTraining accuracy: ", CHD_predictor.score(X_train_scaled, y_train))
print("Testing accuracy: ", CHD_predictor.score(X_test_scaled, y_test))

# predict using model
y_pred = CHD_predictor.predict(X_test_scaled)

# other performance metrics (precision, recall, f1)
print("\n", classification_report(y_test, y_pred))

# confusion matrix, ROC / AUC plotting??