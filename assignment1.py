import pandas as pd
import numpy as np

data = pd.read_csv("train.csv", header = None)
data = data.replace(9999, np.nan)

prior_dict = {}
for label in data.iloc[:,0].unique():  
    prior_dict[label] = data.iloc[:,0].value_counts()[label]/len(data)

mean_array = []
std_array = []
for label in prior_dict.keys():
    categorised_data = data[data.iloc[:,0] == label].iloc[:,1:]
    mean_array.append(categorised_data.mean(axis = 0, skipna= True).values)
    std_array.append(categorised_data.std(axis = 0, skipna = True).values)
mean_array = np.array(mean_array)
std_array = np.array(std_array)
