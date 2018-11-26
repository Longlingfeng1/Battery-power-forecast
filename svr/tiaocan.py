from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier


if __name__ == '__main__':
    raw_data = pd.read_csv("../data/train2.csv")
    raw_data.head()
    train_target = raw_data['charge_energy']
    # print(raw_data.describe())
    cols = ['charge_soc', 'charge_U', 'charge_I', 'charge_temp', 'charge_time', 'charge_end_U']  # 1 3 4 5
    print("-----------------------")
    train_data = raw_data[cols]
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.33, random_state=0)
    #cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
   # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 4, 'min_child_weight': 4, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))