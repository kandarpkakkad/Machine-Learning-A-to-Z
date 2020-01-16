import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
We are human resource team working for a big company and we are about to hire a new employee in this company. This new employee seems to great and good fit for the job and we are about to make a offer to this potential new employee. And now its time to negotiate; negotiate on what is goiing to be the future salary of this new employee. At the beginning of the negotiation this new employee is telling that he's had 20 plus years of experience and eventually earned 160k annual salary in his previous company. So this employee is asking for atleast more than 160k. Hovever there is someone in the HR team that is kind of a control freak and always fantasised about being a detective. So suddenly he decides to call the previous employer to check that info. But unfortunately all the info that this person manages to get are the info in the data set. So the HR member of the team runs a simple analysis of the data in excel and observes that there a non-linear relationship between the position levels and their associated salaries. However this HR person could get aa very relevent information that this new employee has been a region manager for 2 years now and usually it takes on an average 4 years to jump from a regional manager to a partner.

Now this HR guy is getting all excited because he's telling to the team that he can build a bluffing detector using regression models and predict if this new employee is bluffing about his salary.
"""

# Loading dataset

dataset = pd.read_csv("Position_Salaries.csv")
#print(dataset)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Creating training and testing datasets
"""
We here will not create training and test datasets because if the model does not get enough data to train on this model can fail. 
And so this time we can allow the model to train on complete data set
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3)
"""

# Fitting linear regression to the dataset

lregr = LinearRegression()
lregr.fit(x,y)

# Fitting polynomial regression to the dataset

pregr = PolynomialFeatures(degree=12)
x_poly = pregr.fit_transform(x)

# Linear regression model 2 to include the fit using x_poly into our linear regression model

lregr2 = LinearRegression()
lregr2.fit(x_poly,y)

# Visualising the linear regression results

plt.scatter(x,y, color = 'green')
plt.plot(x, lregr.predict(x), color = 'blue')
plt.title("Truth or Bluf(LR)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Visualising the polynomial regression results

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'orange')
plt.plot(x_grid, lregr2.predict(pregr.fit_transform(x_grid)), color = 'purple')
plt.title("Truth or Bluf(PR)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with lineaar regresssion

ma = []
n = []
n.append(6.5)
ma.append(n)
lregr.predict(ma)

# Predicting a new result with polynomial regression

lregr2.predict(pregr.fit_transform(ma))
