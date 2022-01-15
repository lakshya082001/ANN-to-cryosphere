# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset ,when index is considered
dataset1 = pd.read_csv('smb_all_elevations_dataset.csv')

#splitted the dataset to a certain elevation for prediction.
use_cols = ["SMB_6175","Temp_6175", "Prep_6175"]
dataset = pd.read_csv('smb_all_elevations_dataset.csv',usecols=use_cols)

#when the dataframe we are using for prediction contains the date
#dataset.drop(["Date"], axis = 1, inplace = True)

# predicting dataset
use_cols_2 = ["Date", "SMB_6175","Temp_6175", "Prep_6175"]
dataset_2 = pd.read_csv('prediction_all_elevation_dataset.csv',usecols=use_cols_2,index_col=['Date'])
data_2 = dataset_2.values


# ##dataset with date as index

#dataset=pd.read_csv('smb dataset2.csv',index_col=['Date'])

# splitting in X and y
data = dataset.values
X = data[:,1:3]
y = data[:, 0]
# splitting in X and y
x = data_2[:, 1:3]
y_2 = data_2[:, 0]

# checking the index of max value  
#for i in range (0,len(y)):
#    if(y[i]==(Max_)):
#        k=i
#Max_ = max(y)
e=10
y_2=y_2*(e**4)
y=y*(e**4)

# MIN MAX SCALING
#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(dataset)
#scaled_datay_graph = pd.DataFrame({'SMB': dataset[:,2]})

# STANDARD SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X = sc.fit_transform(X)
sc.fit(X)
X = sc.transform(X)
x_2 = sc.transform(x)

sc2 = StandardScaler()
y = np.reshape(y, (y.shape[0], 1))
y = sc2.fit_transform(y)

# ygraph for graph plotting when we have not taken date as indexing 
#y_graph = pd.DataFrame({'SMB': data[:,2]})

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(units = 20, init = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
classifier.add(Dense(units = 20, init = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, nb_epoch = 40)
classifier.fit(X_train, y_train, batch_size = 1, nb_epoch = 20)

# Part 3 - Making the predictions and evaluating the modelimport keras

# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred = sc2.inverse_transform(y_pred)
y_test = sc2.inverse_transform(y_test)


import math
from sklearn.metrics import mean_squared_error
from math import sqrt

##NOTE:there is no significance of calculating rmse therefore we are using R^2 score

#rmse = sqrt(mean_squared_error(y_test,y_pred))
#rmse
for i in range (0,len(y_pred)):
    if(-2<=y_pred[i,0]<=2):
        y_pred[i,0]=0
        
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
        
for i in range (0,len(y_test)):
    if(-1<=y_test[i,0]<=1):
              y_test[i,0]=0
        
        y_pred=y_pred/(e**4)
        y_test=y_test/(e**4)
       y_test
       
        
#mse = mean_squared_error(y_test,y_pred)
#rmse2 = math.sqrt(mse)

#rmse3 = np.sqrt(np.mean((y_pred - y_test)**2 ))
#rmse3

plt.style.use('fivethirtyeight')
training_data_len = math.ceil( (len(data))-(len(data)*0.2+1) )
training_data_len

# plotting the cross validate dataset ,where problem is that we can't plot against date due to random splitting.
valid = dataset[training_data_len:]
valid['original'] = y_test
valid['Predictions'] = y_pred
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('SMB', fontsize=18)
#plt.plot(train['SMB'])
plt.plot(valid[["original",'Predictions']])
plt.legend([ 'real', 'y_pred'], loc='lower right')
plt.show()



### Plotting the predicting dataset ranging from 26/4/1994 to 2016, data is plotted against date.
#predicting values
y_pred2 = classifier.predict(x_2)
y_pred2 = sc2.inverse_transform(y_pred2)
for i in range (0,len(y_pred2)):
    if(-2<=y_pred2[i,0]<=2):
        y_pred2[i,0]=0
r2_score(y_2, y_pred2)

        
y_pred2=y_pred2/(e**2)
y_2=y_2/(e**2)
training_data_len2 = math.ceil( (len(data_2))-(len(data_2)*0.95+1) )
training_data_len2

plt.style.use('fivethirtyeight')
valid = dataset_2[:training_data_len2]
#valid['original'] = y_test
valid['Predictions'] = y_pred2[0:training_data_len2]
valid['original'] = y_2[0:training_data_len2]
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('SMB', fontsize=18)
#plt.plot(train['SMB'])
plt.plot(valid[['original','Predictions']])
plt.legend([ 'real', 'y_pred'], loc='lower right')
plt.show()
