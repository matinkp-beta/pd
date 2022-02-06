import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import  r2_score

from sklearn.model_selection import train_test_split
from scipy import signal



logging.basicConfig(filename="svr.log",
                            filemode='a',
                            format=' %(asctime)s  %(message)s',
                            
                            level=logging.DEBUG)

cs = sys.argv[1]
df = pd.read_csv(f"./cs{cs}.csv")


print(df.shape)
n_coff_decimate = int(input("enter n_coff_decimate: "))
resamle=signal.decimate(df,n_coff_decimate)
df_orgin= pd.DataFrame(resamle, columns = range(len(resamle[0])))
l,s=df_orgin.shape
print(l,s)
y = pd.DataFrame(index=range(l))
y.reset_index(drop=True,inplace=True)

y["x"] = pd.read_csv("./target/x_str.txt",header=None)
y["y"] = pd.read_csv("./target/y_str.txt",header=None)
y["z"] = pd.read_csv("./target/z_str.txt",header=None)
X_train, X_test, y_train, y_test = train_test_split( df_orgin, y, test_size=0.20)

regr = SVR()
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','poly','linear']}



gs_svr = GridSearchCV(regr,param_grid )
clf =MultiOutputRegressor(gs_svr)
clf = clf.fit(X_train,y_train)
print(clf.estimators_[0].best_params_ )

regr = SVR(C=clf.estimators_[0].best_params_["C"],
                    gamma=clf.estimators_[0].best_params_["gamma"],
                    kernel=clf.estimators_[0].best_params_["kernel"]) 
regr =MultiOutputRegressor(regr)
regr = regr.fit(X_train,y_train)

y_pred = regr.predict(X_test)

# The coefficients
#print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
print(regr.score(X_test,y_test))

logging.info(f"\n CS{cs} SVR: {clf.estimators_[0].best_params_ } \n dataset shape: {df.shape} \n Mean squared error: {mean_squared_error(y_test, y_pred)} \n samples: {s} \n --------------------------------- \n")