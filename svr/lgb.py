import sklearn as sk
from sklearn.model_selection import train_test_split
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
df_train=pd.read_csv("../data/train2.csv")
df_train.head()
train_target = df_train['charge_energy']
#print(raw_data.describe())
cols = ['charge_soc','charge_U', 'charge_I','charge_temp', 'charge_time','charge_end_U'] #1 3 4 5
print("-----------------------")
train_data= df_train[cols]
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.33, random_state=0)



# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train) # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据



params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
print('start training...')
#train

'''
gbm = lgb.train(params,
                lgb_train,
               num_boost_round=1,
               valid_sets=lgb_valid,
               early_stopping_rounds=50)'''
gbm = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=lgb_eval, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集
print('Save model...')
gbm.save_model('model.txt')  # 训练后保存模型到文件
print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(y_pred)
print(y_test)
#print(y_pred-y_test)
e=sum(((y_pred-y_test)/y_test)**2)**0.5
print(e)
'''
jieguo = pd.DataFrame()
jieguo['weights']=y_pred
jieguo.to_csv('jieguo.csv',header=None,index=False)
goods_table = pd.read_csv('goods_train.csv',header=None,delimiter='\t')
x_test.columns = ['uid','spu_id','wuyong']
goods_table.columns = ['spu_id','brand_id','cat_id']
df_test = pd.merge(x_test,goods_table,on="spu_id")
df_test.to_csv("df_test.csv")
df_test['newhead']=93
df_test.head()
x_train.head()
jieguo.head()
'''
