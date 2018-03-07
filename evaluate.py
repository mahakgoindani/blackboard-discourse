#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:15:50 2018

@author: csuser
"""

import numpy
from sklearn.metrics import precision_recall_fscore_support

def getStatisticsBinary(values):
    
    numPos = 0.0;
    numNeg = 0.0;
    n = len(values);
    
    for i in range(len(values)):
        if(values[i]=='1'):
            numPos = numPos + 1;
        else:
            numNeg = numNeg + 1;
            
    percentPos = numPos*100.00/n;
    percentNeg = numNeg*100.00/n;
    
    return percentPos, percentNeg


def getScores(Y_test, Y_predict):
    
    FN = 0.0;
    FP = 0.0;
    TP = 0.0;
    TN = 0.0;
    for i in range(len(Y_test)):
        if(Y_test[i] != Y_predict[i]):
            if(Y_predict[i]=='1'):
                FP = FP + 1;
            else:
                FN = FN + 1;
        else:
            if(Y_predict[i]=='1'):
                TP = TP + 1;
            else:
                TN = TN + 1;
    
    #print ('FP, FN, TP, TN')           
    #print (FP, FN, TP, TN)
    
    
    scores = precision_recall_fscore_support(Y_test, Y_predict, average='macro');
    
    return scores[0], scores[1], scores[2], FP, FN, TP, TN
    