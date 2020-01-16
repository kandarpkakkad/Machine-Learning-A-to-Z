import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Loading dataset

dataset = pd.read_csv("Position_Salaries.csv")
#print(dataset)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Creating training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3)


# Fitting the regression model to the dataset

regr = DecisionTreeRegressor(random_state = 0)
regr.fit(x,y)

# Predicting a new result with decision tree regression model

y_pred = regr.predict(np.array([[6.5]]))

"""
# Visualising the decision tree regression model results

plt.scatter(x,y, color = 'orange')
plt.plot(x, regr.predict(x), color = 'purple')
plt.title("Truth or Bluf(Decision Tree Regression Model)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
"""

# Visualising the decision tree regression model results for higher resolution and smoother curve

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color = 'green')
plt.plot(x_grid, regr.predict(x_grid), color = 'red')
plt.title("Truth or Bluf(Regression Model)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
