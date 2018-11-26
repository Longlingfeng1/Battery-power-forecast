import numpy as np
import pandas as pda
data=pda.read_csv("predict_data_e_train.csv")
#data=data[data["vehicle_id"]==5]
data.replace(0, np.nan, inplace=True)#所有零值替代为nan
data=data.fillna(data.median())
#data.to_csv("D:/douban/train.csv",sep=',',index=False)
data["charge_time"]=data["charge_end_time"]-data["charge_start_time"]
data["charge_soc"]=data["charge_end_soc"]-data["charge_start_soc"]
data["charge_U"]=abs(data["charge_end_U"]-data["charge_start_U"])
data["charge_I"]=data["charge_end_I"]-data["charge_start_I"]
data["charge_temp"]=data["charge_max_temp"]-data["charge_min_temp"]

del data["vehicle_id"]
print(data.columns)
data.to_csv("test5.csv", sep=',', index=False)