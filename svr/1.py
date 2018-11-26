from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
raw_data=pd.read_csv("../data/train5.csv")
#raw_data=raw_data[raw_data['charge_soc']>0.1]
#raw_data=raw_data[raw_data['charge_U']>0.1]
train_target = raw_data['charge_energy']
#print(raw_data.describe())
cols = ['charge_soc','charge_U', 'charge_I' , 'charge_temp', 'charge_time'] #1 3 4 5
train_data= raw_data[cols]
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.33, random_state=0)
'''svr_model=SVR()#kernel='linear'
svr_model.fit(X_train,y_train)
result_set_predict=svr_model.predict(X_test)
print(result_set_predict)
y_test=y_test.values
#print(y_test.values)
print(accuracy_score(result_set_predict,y_test))'''
'''
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1145, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_test_pred = forest.predict(X_test)
e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)


#xgboost
'''
model = LGBMRegressor()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
#print(y_test_pred)
#print(y_test_pred)
#print(y_test_pred-y_test)
e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)

