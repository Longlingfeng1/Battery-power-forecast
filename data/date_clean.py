import numpy as np
import pandas as pd


#data1 = data['charge_start_soc'][(data['vehicle_id']==1)]
"""
print('-------------------------------')
data=data[data["vehicle_id"]==4]
print(data)

da=data.values
#print(da)
newdata=data.values
newdata=newdata.T
line=len(newdata)
col=len(newdata[0])
print(line)
print(col)
x=0
#print(newdata[8])
for i in range(0,line):
    for j in range(0,col):
         if(newdata[i][j]==0):
            newdata[i][j]=newdata[i].mean()
            x=x+1
print(newdata)
print("---------------------------")

newdata=newdata.T

"""
#newdf = pda.DataFrame(newdata)
#print(newdf)
#data=pda.read_csv("train.csv")
#data[data["vehicle_id"]==4].replace(newdf)

data=pd.read_csv("predict_data_e_train.csv")

data4=data[data['vehicle_id']==4]
data4.replace(0,np.nan,inplace=True)
data4 = data4.fillna(data4.mean())


data4["charge_time"]=data4["charge_end_time"]-data4["charge_start_time"]
data4["charge_soc"]=data4["charge_end_soc"]-data4["charge_start_soc"]
data4["charge_U"]=data4["charge_end_U"]-data4["charge_start_U"]
data4["charge_I"]=data4["charge_end_I"]-data4["charge_start_I"]
data4["charge_temp"]=data4["charge_max_temp"]-data4["charge_min_temp"]
del data4["charge_end_time"],data4["charge_start_time"],data4["charge_end_soc"],data4["charge_start_soc"],data4["charge_end_U"]
del data4["charge_start_U"],data4["charge_end_I"],data4["charge_start_I"],data4["charge_max_temp"],data4["charge_min_temp"]
temp_df=data4["charge_energy"]
del data4["charge_energy"],data4["vehicle_id"]
data4["charge_energy"] = temp_df
data4.to_csv("train4.csv", sep=',', index=False)

data.replace(0,np.nan,inplace=True)
data = data.fillna(data.mean())

data["charge_time"]=data["charge_end_time"]-data["charge_start_time"]
data["charge_soc"]=data["charge_end_soc"]-data["charge_start_soc"]
data["charge_U"]=data["charge_end_U"]-data["charge_start_U"]
data["charge_I"]=data["charge_end_I"]-data["charge_start_I"]
data["charge_temp"]=data["charge_max_temp"]-data["charge_min_temp"]
del data["charge_end_time"],data["charge_start_time"],data["charge_end_soc"],data["charge_start_soc"],data["charge_end_U"]
del data["charge_start_U"],data["charge_end_I"],data["charge_start_I"],data["charge_max_temp"],data["charge_min_temp"]
data.to_csv("train1.csv", sep=',', index=False)
print(data.columns)

