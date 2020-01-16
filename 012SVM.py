import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


class SVM_ML:
    """
    Author: Kandarp Kakakd
    x_train, y_train, x_test, y_test: train and test set for SVM
    kernel:Specifies the kernel type to be used in the algorithm.
           It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
           If none is given, ‘rbf’ will be used.
           If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)
    random_state: The seed of the pseudo random number generator used when shuffling the data for probability estimates.
                  If int, random_state is the seed used by the random number generator;
                  If RandomState instance, random_state is the random number generator;
                  If None, the random number generator is the RandomState instance used by np.random
    """
    def __init__(self,x_train, y_train, x_test, y_test, kernel="rbf", random_state=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.alpha = alpha
        self.kernel = kernel
        self.random_state = random_state


    def SVM_fit_and_predict(self):
        # Fitting SVM to the training dataset

        classifier = SVC(kernel=self.kernel, random_state=self.random_state)
        classifier.fit(self.x_train, self.y_train)

        # Predicting the test set results

        y_pred = classifier.predict(self.x_test)

        # Making the confusion matrix

        cm = confusion_matrix(self.y_test, y_pred)

        # Plotting the SVM Training set curve

        x_set, y_set = self.x_train, self.y_train
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
        plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('blue', 'pink'))(i), label=j)
        plt.title('Classifier (Training set)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

        # Plotting the SVM Testing set curve

        x_set, y_set = x_test, y_test
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
        plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                        c=ListedColormap(('blue', 'pink'))(i), label=j)
        plt.title('Classifier (Testing set)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

# Loading the dataset

dataset = pd.read_csv("Social_Network_Ads.csv")
#print(dataset)

# Dividing the dataset for comarision

x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
#print(x)
#print(y)

# Creating training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
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

# Fitting SVM to the training dataset

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the test set results

y_pred = classifier.predict(x_test)

# Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plotting the SVM Training set curve

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('blue', 'pink'))(i), label=j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Plotting the SVM Testing set curve

x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('blue', 'pink'))(i), label=j)
plt.title('Classifier (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
