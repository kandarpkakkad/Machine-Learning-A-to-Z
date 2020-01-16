import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the dataset

# dataset = pd.read_csv("Data.csv")
# print(dataset)

# Dividing the dataset for comarision

"""
Note:
    x here should be a matrix and y should be a vector
"""

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)

# Adding the missing values by mean(meadian and mode can also be used)

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print(x)

# Converting the string data into numerical data to compare using mathematical equation

le_x = LabelEncoder()
x[:, 0] = le_x.fit_transform(x[:, 0])
ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()

le_y = LabelEncoder()
y = le_y.fit_transform(y)
# print(x)
# print(y)

# Creating training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

"""
Feature Scaling:
    Here the age and salary columns does not have similar numbers.
    The difference is too large that the ML model can neglect age.
    There are 2 methods for feature scaling:
        Standardisation  --> (x-mean(x)) / (std dev(x))
        Normalisation  --> (x-min(x)) / (max(x)-min(x))
"""
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# print(x_train)
# print(x_test)
