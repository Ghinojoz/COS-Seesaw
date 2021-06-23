from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from interpret import show
from interpret import data
from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree
from interpret.perf import RegressionPerf

df = pd.read_pickle('COS_Seesaw_dataframe.pkl')

print(df.dtypes)

columns = list(df.columns)
if 'COS_JFJ' in columns:
    columns.remove('COS_JFJ')
else:
    print(columns)

if 'time' in columns:
    columns.remove('time')

x = df[columns]
y = df['COS_JFJ']
seed = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

print (len(X_train))
print (len(y_train))

print(len(X_test))
print(len(y_test))

print(X_train)
print(y_train)
marginal = data.Marginal(feature_names=columns)
marginal_explanation = marginal.explain_data(X_train, y_train)
show(marginal_explanation)
