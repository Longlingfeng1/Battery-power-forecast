from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
raw_data=pd.read_csv("../data/train2.csv")
raw_data.head()
y_train = raw_data['charge_energy']
#print(raw_data.describe())
cols = ['charge_soc','charge_U', 'charge_I','charge_temp', 'charge_time','mileage']
X_train= raw_data[cols]
data=pd.read_csv("../data/test2.csv")
X_test=data[cols]
#X_test=(X_test-X_test.mean())/X_test.std()
#X_train=(X_train-X_train.mean())/X_train.std()
#y_train=(y_train-y_train.mean())/y_train.std()
#X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.33, random_state=0)

model = XGBClassifier()
model.fit(X_train, y_train)
print(X_test)
y_test_pred = model.predict(X_test)
print(y_test_pred)
