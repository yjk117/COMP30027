import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv", header = None)
train_data = train_data.replace(9999, np.nan)

test_data = pd.read_csv("test.csv", header = None)
test_data = test_data.replace(9999, np.nan)

prior_dict = {}
for label in train_data.iloc[:,0].unique():  
    prior_dict[label] = train_data.iloc[:,0].value_counts()[label]/len(train_data)

mean_array = []
std_array = []
for label in prior_dict.keys():
    categorised_data = train_data[train_data.iloc[:,0] == label].iloc[:,1:]
    mean_array.append(categorised_data.mean(axis = 0, skipna= True).values)
    std_array.append(categorised_data.std(axis = 0, skipna = True, ddof = 0).values)
    
prediction = []
for i in range(0, len(test_data)):
    prob_dict = {}
    for label in prior_dict.keys():
        x = test_data.iloc[i,1:].values.astype(np.float64)
        index = list(prior_dict.keys()).index(label)
        mu = mean_array[index]
        sigma = std_array[index]
        prior = prior_dict[label]
        pdf = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 *((x - mu)/sigma)**2)
        pdf[pdf == 0] = np.nan
        prob_dict[label] = np.log(prior) + np.nansum(np.log(pdf))
    prediction.append(max(prob_dict, key = prob_dict.get))
    
correct = 0
for i in range(len(test_data)):
    if test_data.iloc[i,0] == prediction[i]:
        correct += 1
print(correct/len(test_data))
