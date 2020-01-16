import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd .read_csv("Market_Basket_Optimisation.csv", header = None)
#print(dataset)

transactions = []
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])

# Training apriori on the dataset

rules = apriori(transactions, min_support=0.005, min_confidence=0.5, min_lift=5, min_length=2)

# Visualising the results

result = list(rules)


