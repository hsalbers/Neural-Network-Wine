# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:37:08 2017

@author: Hope
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split

data = pd.read_csv("C:\Users\Hope\Desktop\MSBAFinalScripts\FinalAIScripts\winequality-white.csv",
                   sep = ";", header=int(0))
              
data = data.dropna()
#print data.head() 

dep_vars = ['quality']
indep_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
'alcohol' ]

dep_data = data[dep_vars] 
indep_data = data[indep_vars]
indep_train, indep_test, dep_train, dep_test = train_test_split(indep_data, dep_data, test_size=0.33, random_state=2016)

np.random.seed(2016)

from sklearn import preprocessing
indep_train_MinMax = preprocessing.MinMaxScaler()
indep_test_MinMax = preprocessing.MinMaxScaler()
dep_train_MinMax = preprocessing.MinMaxScaler()
dep_test_MinMax = preprocessing.MinMaxScaler()

indep_test.as_matrix(indep_test)
indep_test = np.array(indep_test).reshape((len(indep_test), 11))
indep_train = np.array(indep_train).reshape((len(indep_train), 11))

dep_test.as_matrix(dep_test)
dep_test = np.array(dep_test).reshape((len(dep_test), 1))
dep_train = np.array(dep_train).reshape((len(dep_train), 1))

indep_train = indep_train_MinMax.fit_transform(indep_train)
indep_test = indep_test_MinMax.fit_transform(indep_test)
dep_train = dep_train_MinMax.fit_transform(dep_train)
dep_test = dep_test_MinMax.fit_transform(dep_test)

indep_train.mean(axis=0)
indep_train.mean(axis=0)

from sknn.mlp import Regressor, Layer
fit1 = Regressor(
        layers=[
                Layer("Sigmoid", units=45),
                     Layer("Sigmoid", units=18),
                          Layer("Sigmoid", units=18),
                               Layer("Linear")],
                learning_rate=0.80,
                random_state=2016,
                valid_size=0.25,
                learning_momentum=0.30,
                n_iter=100)
print "fitting model now"
fit1.fit(indep_train, dep_train)
pred2_train=fit1.predict(indep_train)
mse_2 = mean_squared_error(pred2_train, dep_train)
mae_2 = mean_absolute_error(pred2_train, dep_train)
        
pred4_test=fit1.predict(indep_test)
mse_4 = mean_squared_error(pred4_test,dep_test)
mae_4 = mean_absolute_error(pred4_test,dep_test)
print "MSE train = ", mse_2
print "MSE test = ", mse_4

print "MAE train = ", mae_2 
print "MAE test = ", mae_4 