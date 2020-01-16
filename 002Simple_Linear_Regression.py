import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math as m

#Data Preprocessing

dataset = pd.read_csv("Salary_Data.csv")
#print(dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 1/3)

#Fitting simple linear regression to the training set

regr = LinearRegression()
regr.fit(x_train, y_train)

#Predicting the test set results

y_pred = regr.predict(x_test)

#Plotting the graphs foor comparision

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regr.predict(x_train), color='blue')
plt.title("Salary vs Experience(Training)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, color='green')
plt.plot(x_test,y_pred, color='purple')
plt.title("Salary vs Experience(Testing)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
