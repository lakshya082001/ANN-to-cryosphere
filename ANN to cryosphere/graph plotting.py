import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
plt.style.use('fivethirtyeight')
# Importing the dataset

## 4075 means original value at 4075m
## 4076 means predicted value at 4075m

#s=["4075","4076",	"4125","4126"	,"4175"	,"4176",	"4225"	,"4226",	"4275",	"4276",	"4325"	,"4326"	,"4375",	"4376",	"4425",	"4426"	,"4475"	,"4476"	,"4525"	,"4526"	,"4575"	,"4576",	"4625",	"4626",	"4675"	,"4676"	,"4725"	,"4726",	"4775"	,"4776"	,"4825"	,"4826",	"4875",	"4876",	"4925"	,"4926",	"4975"	,"4976"	,"5025"	,"5026",	"5075",	"5076",	"5125"	,"5126"	,"5175",	"5176",	"5225"	,"5226"	,"5275"	,"5276"	,"5325"	,"5326",	"5375"	,"5376"	,"5425",	"5426",	"5475"	,"5476",	"5525"	,"5526"	,"5575", "5576"	,"5625",	"5626",	"5675",	"5676",	"5725"	,"5726",	"5775"	,"5776"	,"5825",	"5826"	,"5875",	"5876"	,"5925"	,"5926",	"5975"	,"5976"	,"6025"	,"6026"	,"6075",	"6076"	,"6125"	,"6126"  ,"6175"	,"6176"
#]
dataset_1 = pd.read_csv('Yearly averaged data.csv',index_col=['Year'])

for i in range (0,43):
    a=(2*i)+1
    b=(2*i)+2
    filename = str(4075+(50*i)) +'.png'
    Title = str(4075+(50*i)) +' elevation SMBs'

    use_cols = [0,a,b]
        
    dataset = pd.read_csv('Yearly averaged data.csv',usecols=use_cols,index_col=['Year'])
    data = dataset.values
    #dataframe = data[0:800 , :]
    #dataframe = pd.DataFrame({'predictions', 'original SMB': dataframe[:,0:2]})
    #data = data/(10**4)
    
    
    #training_data_len = math.ceil((len(data))
    #training_data_len
    
    
    #train = y_graph[:training_data_len]
    valid = dataset
    valid['original'] = data[:,0]
    valid['prediction'] = data[:,1]
    #Visualize the data
    fig=plt.figure(figsize=(16,8))
    plt.title(Title)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('SMB', fontsize=18)
    #plt.plot(train['SMB'])
    plt.plot(valid[["original",'prediction']])
    plt.legend(['original', 'predicted'], loc='lower right')
    plt.savefig(filename)
    plt.close(fig)

###############################################################################################
#### for glacier wide
    
use_cols2 = ["Year","Predicted Glacier wide values","Original Glacier Wide SMB values"]
dataset = pd.read_csv('Yearly averaged data.csv',usecols=use_cols2,index_col=['Year'])
data = dataset.values

#from sklearn.metrics import r2_score
#y_test = data[0:22,0]
#y_pred = data[0:22,1]
#r2_score(y_test, y_pred)

valid = dataset
valid['original'] = data[:,1]
valid['prediction'] = data[:,0]
#Visualize the data
fig=plt.figure(figsize=(16,8))
plt.title('Glacier wide SMB vs year')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Glacier wide SMB', fontsize=18)
    #plt.plot(train['SMB'])
plt.plot(valid[["original",'prediction']])
plt.legend(['original', 'predicted'], loc='lower right')
plt.savefig('final_averaged_single_plot.png')
plt.close(fig)
    

#### for Mean SMB variation with elevation
use_cols2 = ["Elevations","Mean orig SMB","Mean pred SMB"]
dataset = pd.read_csv('Yearly averaged data.csv',usecols=use_cols2,index_col=['Elevations'])
data = dataset.values
valid = dataset
valid['original'] = data[:,0]
valid['prediction'] = data[:,1]
#Visualize the data
fig=plt.figure(figsize=(16,8))
plt.title('Mean SMBs vs elevation')
plt.xlabel('Elevations', fontsize=18)
plt.ylabel('Mean SMB', fontsize=18)
    #plt.plot(train['SMB'])
plt.plot(valid[["original",'prediction']])
plt.legend(['original', 'predicted'], loc='lower right')
plt.savefig('Mean SMB vs Elevations.png')
plt.close(fig)

























