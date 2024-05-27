import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.metrics import root_mean_squared_error

#the start and end date
start_date = dt.datetime(1995,3,1)
end_date = dt.datetime(2016,11,10)
ep = 50

#loading from yahoo finance
data = pd.read_csv("TSLA.csv")
data.set_index("DATE",inplace=True)

#data = data[:math.ceil(len(data)*0.7)].iloc[:,:1]
print(data.shape)

# Setting 80 percent data for training
training_data_len = math.ceil(len(data) * 0.9)
#Splitting the dataset
train_data = data[:training_data_len].iloc[:,:1] 
test_data = data[training_data_len:].iloc[:,:1]

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
	X_train.append(scaled_train[i-50:i-12, 0])
	y_train.append(scaled_train[i, 0])


X_test = []
y_test = []
for i in range(50, len(scaled_test)):
	X_test.append(scaled_test[i-50:i-12, 0])
	y_test.append(scaled_test[i, 0])
    


# The data is converted to Numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
print("X_train :",X_train.shape,"y_train :",y_train.shape)

# The data is converted to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)
#Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
print("X_test :",X_test.shape,"y_test :",y_test.shape)

# importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# initializing the RNN
regressor = Sequential()
# adding RNN layers and dropout regularization
regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True,
						input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True))
regressor.add(SimpleRNN(units = 50,
						activation = "tanh",
						return_sequences = True))
regressor.add( SimpleRNN(units = 50))
# adding the output layer
regressor.add(Dense(units = 1,activation='sigmoid'))
# compiling RNN
regressor.compile(optimizer = SGD(learning_rate=0.01,
								decay=1e-6, 
								momentum=0.9, 
								nesterov=True), 
				loss = "mean_squared_error")
# fitting the model
regressor.fit(X_train, y_train, epochs = ep, batch_size = 2)
regressor.summary()


#Initialising the model
regressorLSTM = Sequential()
#Adding LSTM layers
regressorLSTM.add(LSTM(50, 
					return_sequences = True, 
					input_shape = (X_train.shape[1],1)))
regressorLSTM.add(LSTM(50, 
					return_sequences = False))
regressorLSTM.add(Dense(25))
regressorLSTM.add(Dense(25))
regressorLSTM.add(Dense(25))
#Adding the output layer
regressorLSTM.add(Dense(1))
#Compiling the model
regressorLSTM.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy"])
#Fitting the model
regressorLSTM.fit(X_train, 
				y_train,  
				epochs = ep)
regressorLSTM.summary()



# predictions with X_test data
y_RNN = regressor.predict(X_test)
y_LSTM = regressorLSTM.predict(X_test)


# scaling back from 0-1 to original
y_RNN_O = scaler.inverse_transform(y_RNN) 
y_LSTM_O = scaler.inverse_transform(y_LSTM) 


fig, axs = plt.subplots(2,figsize =(50,30),sharex=True, sharey=True)
fig.suptitle('Model Predictions')

#Plot for RNN predictions
axs[0].plot(train_data.index[:], train_data[:], label = "train_data", color = "b")
axs[0].plot(test_data.index, test_data, label = "test_data", color = "g")
axs[0].plot(test_data.index[50:], y_RNN_O, label = "y_RNN", color = "brown")
axs[0].legend()
axs[0].title.set_text("Basic RNN")

#Plot for LSTM predictions
axs[1].plot(train_data.index[:], train_data[:], label = "train_data", color = "b")
axs[1].plot(test_data.index, test_data, label = "test_data", color = "g")
axs[1].plot(test_data.index[50:], y_LSTM_O, label = "y_LSTM", color = "orange")
axs[1].legend()
axs[1].title.set_text("LSTM")


plt.xlabel("Days")
plt.ylabel("Open price")

plt.show()

print(root_mean_squared_error(test_data[50:],y_RNN_O))
print(root_mean_squared_error(test_data[50:],y_LSTM_O))