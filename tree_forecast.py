import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error
from scipy import stats
from scipy.special import boxcox, inv_boxcox


#the start and end date
start_date = dt.datetime(1995,3,1)
end_date = dt.datetime(2016,11,10)


#loading from yahoo finance
data = pd.read_csv("TSLA.csv")
data.set_index("DATE",inplace=True)
#data = data[:math.ceil(len(data)*0.7)].iloc[:,:1]
# Setting 80 percent data for training
training_data_len = math.ceil(len(data) * 0.9)
#Splitting the dataset
train_data = data[:training_data_len].iloc[:,:1] 
test_data = data[training_data_len:].iloc[:,:1]
train_data["VALUE"],dlambda_train = stats.boxcox(train_data["VALUE"])
test_data["VALUE"],dlambda_test = stats.boxcox(test_data["VALUE"])

dataset_train = np.reshape(train_data, (-1,1)) 
dataset_test = np.reshape(test_data, (-1,1)) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)
# Normalizing values between 0 and 1
scaled_test = scaler.fit_transform(dataset_test)  

X_train = []
y_train = []
for i in range(50, len(scaled_train)):
	X_train.append(scaled_train[i-21:i-12, 0])
	y_train.append(scaled_train[i, 0])


X_test = []
y_test = []
for i in range(50, len(scaled_test)):
	X_test.append(scaled_test[i-21:i-12, 0])
	y_test.append(scaled_test[i, 0])
    


# The data is converted to Numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0]))
print("X_train :",X_train.shape,"y_train :",y_train.shape)

# The data is converted to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)
#Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0]))
print("X_test :",X_test.shape,"y_test :",y_test.shape)

regr = RandomForestRegressor(max_depth=5, random_state=0)
regr.fit(X_train, y_train)
y_pre = regr.predict(X_test)
y_pred = scaler.inverse_transform(np.reshape(y_pre,(y_pre.shape[0],1) ))
y_train = scaler.inverse_transform(np.reshape(y_train,(y_train.shape[0],1) ))
y_test = scaler.inverse_transform(np.reshape(y_test,(y_test.shape[0],1) ))

y_pred = inv_boxcox(y_pred,dlambda_test)
y_test = inv_boxcox(y_test,dlambda_test)
train_data = inv_boxcox(train_data,dlambda_train)
test_data = inv_boxcox(test_data,dlambda_test)

plt.figure(figsize=(30,20))
plt.plot(train_data, color = "b")
plt.plot(test_data.index,test_data,color="g")
plt.plot(test_data.index[50:] ,y_pred, color="r")

print(root_mean_squared_error(y_test,y_pred))