import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
raw_data=pd.read_csv("../data/train4.csv")
#raw_data=raw_data[raw_data['charge_soc']>0.1]
data=raw_data[raw_data['charge_U']>=10]
train_target = data['charge_energy'].values
#print(raw_data.describe())
cols = ['charge_soc','charge_U', 'charge_I','charge_temp', 'charge_time'] #1 3 4 5
train_data= data[cols].values
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.33, random_state=42)
'''svr_model=SVR()#kernel='linear'
svr_model.fit(X_train,y_train)
result_set_predict=svr_model.predict(X_test)
print(result_set_predict)
y_test=y_test.values
#print(y_test.values)
print(accuracy_score(result_set_predict,y_test))'''

forest = RandomForestRegressor(n_estimators=1000,n_jobs = -1)
forest.fit(X_train, y_train)
y_test_pred = forest.predict(X_test)
e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)



#xgboost
'''
model = XGBClassifier()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
#print(y_test_pred)
#print(y_test_pred)
#print(y_test_pred-y_test)
e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)
'''