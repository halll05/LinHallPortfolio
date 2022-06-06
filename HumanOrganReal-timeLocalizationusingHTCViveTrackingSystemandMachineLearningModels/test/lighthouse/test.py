import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np


df = pd.read_csv('4-combined_csv.csv') 
print('check if y has strong connections with your X:')
print(df.corr())

# you can choose other models
# try GBR https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# try SVR
reg = linear_model.Ridge(alpha=.5)

y = df.values[:,0]
X = df.values[:,1:]

print("cross validation score is displayed below:")
print(np.average(cross_val_score(reg, X, y, cv=5)))
