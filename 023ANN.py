# Part 1 Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Loading the dataset

dataset = pd.read_csv("Churn_Modelling.csv")
#print(dataset)

# Dividing the dataset for comarision

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#print(x)
#print(y)

# Encoding the categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x_1 = LabelEncoder()
x[:, 1] = le_x_1.fit_transform(x[:, 1])
le_x_2 = LabelEncoder()
x[:, 2] = le_x_2.fit_transform(x[:, 2])
ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()


# Creating training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)


# Feature Scaling

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#print(x_train)
#print(x_test)

# Part 2 Making of ANN

# Importing the libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and hidden layer

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=12))

# Adding second hidden layer

classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set

classifier.fit(x_train, y_train, batch_size=12, nb_epoch=100)

# Part 3 Making the prediction and evaluating the model

# Predicting the test set results

y_pred = classifier.predict(x_test)
y_pred = (y_pred >= 0.6)

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
