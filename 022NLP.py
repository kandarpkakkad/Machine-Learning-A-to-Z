import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Importing the data set

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
#print(dataset)

# Cleaning the data set

corpus = []
ps = PorterStemmer()
nltk.download('stopwords')
for i in range(0, dataset['Review'].size):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of Words model

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Creating training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)

# Fitting Naive Bayes to the training dataset

classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results

y_pred = classifier.predict(x_test)

# Making the confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
