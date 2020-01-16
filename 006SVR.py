import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Loading dataset

dataset = pd.read_csv("Position_Salaries.csv")
#print(dataset)
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Creating training and testing datasets
"""
We here will not create training and test datasets because if the model does not get enough data to train on this model can fail. 
And so this time we can allow the model to train on complete data set
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3)
"""

# Feaature Scaling

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
#print(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
#print(y)

# Fitting the regression model to the dataset

regr = SVR(kernel = 'rbf')
regr.fit(x, y)

# Predicting a new result with SVR

y_pred = sc_y.inverse_transform(regr.predict(sc_x.fit_transform(np.array([[6.5]]))))

# Visualising the SVR results

plt.scatter(x,y, color = 'orange')
plt.plot(x, regr.predict(x), color = 'purple')
plt.title("Truth or Bluf(Regression Model)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
