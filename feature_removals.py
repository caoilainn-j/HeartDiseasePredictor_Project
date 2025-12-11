# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#
#  Prepping data set for accuracy as is
#

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

#
#  End of prepping the data set
#



#
#  Starting Process of Elimination for each feature
#

# Quick method to get the logisitc regression of the passed data frame
# Will be called after removing features for quick access to the accuracy after modifying data
def logisticRegressionTest(df):
    # create test-train-split, accounting for imbalanced dataset using stratify=y (maybe look into SMOTE?)
    y = df[['TenYearCHD']]
    X = df.drop(['TenYearCHD'], axis=1)
    
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
    print(f'\nTraining accuracy: {100 * CHD_predictor.score(X_train_scaled, y_train): .2f}%')
    print(f'Testing accuracy: {100 * CHD_predictor.score(X_test_scaled, y_test): .2f}%')

# Testing to make sure the precision still works as intended
logisticRegressionTest(heart_data)

# ******* BPMeds Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['BPMeds'].copy()

# Seeing results from removal
heart_data.drop('BPMeds', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* prevalentStroke Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['prevalentStroke'].copy()

# Seeing results from removal
heart_data.drop('prevalentStroke', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* prevalentHyp Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['prevalentHyp'].copy()

# Seeing results from removal
heart_data.drop('prevalentHyp', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['prevalentHyp'] = col
logisticRegressionTest(heart_data)

# ******* male Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['male'].copy()

# Seeing results from removal
heart_data.drop('male', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['male'] = col
logisticRegressionTest(heart_data)

# ******* education Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['education'].copy()

# Seeing results from removal
heart_data.drop('education', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* education Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['BMI'].copy()

# Seeing results from removal
heart_data.drop('BMI', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* age Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['age'].copy()

# Seeing results from removal
heart_data.drop('age', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['age'] = col
logisticRegressionTest(heart_data)

# ******* cigsPerDay Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['cigsPerDay'].copy()

# Seeing results from removal
heart_data.drop('cigsPerDay', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['cigsPerDay'] = col
logisticRegressionTest(heart_data)

# ******* currentSmoker Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['currentSmoker'].copy()

# Seeing results from removal
heart_data.drop('currentSmoker', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* diabetes Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['diabetes'].copy()

# Seeing results from removal
heart_data.drop('diabetes', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* TotChol Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['totChol'].copy()

# Seeing results from removal
heart_data.drop('totChol', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# ******* sysBP Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['sysBP'].copy()

# Seeing results from removal
heart_data.drop('sysBP', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['sysBP'] = col
logisticRegressionTest(heart_data)

# ******* diaBP Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['diaBP'].copy()

# Seeing results from removal
heart_data.drop('diaBP', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['diaBP'] = col
logisticRegressionTest(heart_data)

# ******* heartRate Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['heartRate'].copy()

# Seeing results from removal
heart_data.drop('heartRate', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['heartRate'] = col
logisticRegressionTest(heart_data)

# ******* glucose Removal *******
# Saving the column in case we need to add it back after removing it
col = heart_data['glucose'].copy()

# Seeing results from removal
heart_data.drop('glucose', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# Add feature back into data set
heart_data['glucose'] = col
logisticRegressionTest(heart_data)

#
#  End of Process of Elimination
#



#
#  Process of Removing Group of Features all at once
#

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

# Check accuracy at the beginning before removal
logisticRegressionTest(heart_data)

# Remove the group of features
heart_data.drop('education', axis = 1, inplace = True)
heart_data.drop('currentSmoker', axis = 1, inplace = True)
heart_data.drop('BPMeds', axis = 1, inplace = True)
heart_data.drop('prevalentStroke', axis = 1, inplace = True)
heart_data.drop('diabetes', axis = 1, inplace = True)
heart_data.drop('totChol', axis = 1, inplace = True)
heart_data.drop('BMI', axis = 1, inplace = True)
logisticRegressionTest(heart_data)

# check columns after removal
print(heart_data.columns)

#
#  End of Process of Removing Group
#
