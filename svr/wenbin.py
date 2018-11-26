import pandas as pda
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
train_data = pda.read_csv("../data/train4.csv")
data4 = train_data[train_data['charge_U']>=1]
#data4 = train_data[train_data['charge_I']>=-200]
cols = ['charge_soc','charge_U','charge_I','charge_temp','charge_time']
x = data4[cols].values
y = data4['charge_energy'].values

X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=1)
clf = RandomForestRegressor(n_estimators=1000,n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(sum(((y_pred - y_test)/(y_test))**2)**0.5)
