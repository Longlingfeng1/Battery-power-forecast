
import lightgbm as lgb
import xgboost as xb
import sklearn
import time
#import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_csv('../data/train4.csv').values
x = raw_data[0:1200, :-1]  # 分割自变量
y = raw_data[0:1200, -1]  # 分割因变量
#x=(x-x.mean())/x.std()
svr_model=SVR(C=1024,gamma=0.5)
svr_model.fit(x,y)
xx=raw_data[:,:-1]
yy=raw_data[:, -1]
result_set_predict=svr_model.predict(xx)
print(result_set_predict)
print(yy)
e=sum(((result_set_predict-yy)/yy)**2)**0.5
print(e)