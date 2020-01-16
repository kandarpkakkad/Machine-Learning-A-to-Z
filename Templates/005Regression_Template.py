import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

# Loading dataset

#dataset = pd.read_csv("Position_Salaries.csv")
#print(dataset)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Creating training and testing datasets
"""
We here will not create training and test datasets because if the model does not get enough data to train on this model can fail. 
And so this time we can allow the model to train on complete data set
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3)
"""

"""
Feature Scaling:
    Here the age and salary columns does not have similar numbers.
    The difference is too large that the ML model can neglect age.
    There are 2 methods for feature scaling:
        Standardisation  --> (x-mean(x)) / (std dev(x))
        Normalisation  --> (x-min(x)) / (max(x)-min(x))
        
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
"""

# Fitting the regression model to the dataset

# Create your regression here.

# Predicting a new result with regression model

y_pred = regr.predict(np.array([[6.5]]))

# Visualising the regression model results

plt.scatter(x,y, color = 'orange')
plt.plot(x, regr.predict(x), color = 'purple')
plt.title("Truth or Bluf(Regression Model)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Visualising the regression model results for higher resolution and smoother curve

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color = 'green')
plt.plot(x_grid, regr.predict(x_grid), color = 'red')
plt.title("Truth or Bluf(Regression Model)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
