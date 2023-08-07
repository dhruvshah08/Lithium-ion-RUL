"""
SVM - RUL Prediction of Lithium Ion Batteries
"""

import pandas as pd
import numpy as np

#Loading the data
dataset=pd.read_csv('Input n Capacity.csv')

#Formatting the data
dataset = dataset.drop(labels=['SampleId'], axis=1)
data = dataset[~dataset.isin(['?'])]
data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)
print (data.dtypes)
print(data.count())

#Distribution of attributes
CycleC = data[1:200]
CycleC.plot( x='Cycle', y='Capacity(Ah)', color='purple', label='Cycle Capacity')

#Splitting the data into X & Y
X = np.array(data.iloc[:,0:5].values)
y = np.array(data.iloc[:,5].values)

#Dividing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train)
print (x_train)

#Creating the model
from sklearn import svm
parameters = [{'C': [0.001, 0.1 ,1,5], 'kernel':['linear']},{'C': [0.001, 0.1, 1,5], 'kernel':['rbf'], 'gamma':[0.01, 0.05]}]
classifier = svm.SVC(gamma=0.001, C= 0.001, kernel='linear')
#classifier = SVC(kernel='rbf', random_state = 1)
print(classifier)

#Encoding the y dataset
from sklearn import metrics, svm
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_encoded = lab_enc.fit_transform(y_train)
print(y_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(y_encoded))
print(y_encoded.shape)

#Fitting it to the testing and training data
classifier.fit(x_train, y_encoded)

#Predicting the values
Y_pred = classifier.predict(x_test)
Y_pred.shape
print(Y_pred)