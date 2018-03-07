#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:08:18 2018

@author: csuser
"""

import pickle
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold


def runSVM(X_train, Y_train, X_test, Y_test):
    
    svc = SVC(C=1, kernel='linear')
    skf = StratifiedKFold(n_splits=5)
    print (cross_val_score(svc, X_train, Y_train, cv=skf, n_jobs=-1, scoring='f1_macro'))
    svc.fit(X_train, Y_train);
    Y_predict = svc.predict(X_test)
    #print (Y_predict)
        
    return Y_predict;