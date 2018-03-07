#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:08:00 2018

@author: csuser
"""

import numpy


X1 = X[:,[0,1]]
X2 = X[:,[0,2]]
X3 = X[:,[1,2]]

X_train1, Y_train1, X_test1, Y_test1 = getTrainAndTestData(filenumber_index_mapping, X1, fileName_Labels, trainingFilenumbers, testFilenumbers)
print ('X1')
Y_predict1 = runSVM(X_train1, Y_train1, X_test1, Y_test1); 
result = getScores(Y_test1, Y_predict1)
print (result)

X_train2, Y_train2, X_test2, Y_test2 = getTrainAndTestData(filenumber_index_mapping, X2, fileName_Labels, trainingFilenumbers, testFilenumbers)
print ('X2')
Y_predict2 = runSVM(X_train2, Y_train2, X_test2, Y_test2);
result = getScores(Y_test2, Y_predict2)
print (result)

X_train3, Y_train3, X_test3, Y_test3 = getTrainAndTestData(filenumber_index_mapping, X3, fileName_Labels, trainingFilenumbers, testFilenumbers)
print ('X3')
Y_predict3 = runSVM(X_train3, Y_train3, X_test3, Y_test3);
result = getScores(Y_test3, Y_predict3)
print (result)