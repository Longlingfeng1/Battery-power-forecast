import numpy as np
import pandas as pd

data=pd.read_csv("predict_data_e_train.csv")

data.replace(0,np.nan,inplace=True)
data = data.fillna(data.mean())
data["charge_time"]=data["charge_end_time"]-data["charge_start_time"]
data["charge_soc"]=data["charge_end_soc"]-data["charge_start_soc"]
data["charge_U"]=data["charge_end_U"]-data["charge_start_U"]
data["charge_I"]=data["charge_end_I"]-data["charge_start_I"]
data["charge_temp"]=data["charge_max_temp"]-data["charge_min_temp"]
data.to_csv("train_full.csv", sep=',', index=False)
