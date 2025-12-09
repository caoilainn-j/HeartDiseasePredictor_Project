# exploration of data, cleaning up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# look at number of instances that have ten-year risk of coronary heart disease
sns.countplot(data=heart_data, x='TenYearCHD')
plt.show() # notes an imbalanced dataset, need to take steps to ensure testing and training receive same portion

# plot heatmap to look at correlations between variables (look primarily at TenYearCHD)
sns.heatmap(heart_data.corr(), annot=True)
plt.show()