from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from lightgbm import LGBMRegressor
#import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

raw_data=pd.read_csv("../data/train1.csv")
raw_data = raw_data[raw_data['charge_U']>=10]
raw_data.head()
train_target = raw_data['charge_energy']
#print(raw_data.describe())
cols = ['charge_soc','charge_U', 'charge_I','charge_temp', 'charge_time'] #1 3 4 5
print("-----------------------")

train_data= raw_data[cols]
x = raw_data[cols].values
y = raw_data['charge_energy'].values
#x_scaled = preprocessing.StandardScaler().fit_transform(x)
#y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1, 1))
#train_data=(train_data-train_data.mean())/train_data.std()
#train_target=(train_target-train_target.mean())/train_target.std()

X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.33, random_state=0)

'''svr_model=SVR()#kernel='linear'
svr_model.fit(X_train,y_train)
result_set_predict=svr_model.predict(X_test)
print(result_set_predict)
y_test=y_test.values
#print(y_test.values)
print(accuracy_score(result_set_predict,y_test))'''


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion='mse',  random_state=1,n_jobs=-1)
forest.fit(X_train, y_train)
y_test_pred = forest.predict(X_test)
e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)
print("-----------------------")

#xgboost
"""
model = LGBMRegressor()
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

#print(y_test_pred)

e=sum(((y_test_pred-y_test)/y_test)**2)**0.5
print(e)

print('Start training...')
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.1,n_estimators=500)
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='l1',early_stopping_rounds=5)

print('Start predicting...')
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
e=sum(((y_pred-y_test)/y_test)**2)**0.5

print('The rmse of prediction is:', e)

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [600,700,800]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)
"""
#print('Best parameters found by grid search are:', gbm.best_params_)
