#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:12:30 2018

@author: csuser
"""

from svm_model import runSVM
import features
import sys
import numpy as np
from data import getData
from data import getLabels
from data import loadObject
from data import storeObject
from data import getTrainAndTestData
from data import getFilenumberIndexMapping
from data import getTrainAndTestFiles
from data import getFoldsTrainAndTestFiles
from data import getBinaryLabels
from bag_of_words_model import runBagOfWords
from evaluate import getStatisticsBinary
from evaluate import getScores


def main():
    
    directory = '../Blackboard/Depth2PlusUnder10k/'
    
    #fileName_Contents = getData(directory);
    file_Name = '../Blackboard/' + 'Depth2PlusUnder10kDataDump'
    #storeObject(fileName_Contents, file_Name); 
    fileName_Contents = loadObject(file_Name);
    
    labelsFileName = '../Blackboard/' + 'Depth2PlusUnder10kLabels.csv'
    #fileName_Labels = getLabels(labelsFileName);
    labelsFileNameDump = '../Blackboard/' + 'Depth2PlusUnder10kLabelsDump'
    #storeObject(fileName_Labels, labelsFileNameDump); 
    fileName_Labels = loadObject(labelsFileNameDump);
    
    print(fileName_Contents.keys()==fileName_Labels.keys())
    #sys.exit(0);
        
    #print (fileName_Contents)
        
    averageDepth = features.findAverageDepth(fileName_Contents);
    averageNumberOfPosts = features.findAverageNumberOfPosts(fileName_Contents);
    averageNumberOfWords = features.findAverageNumberOfWords(fileName_Contents);
    
    file_parent_children, file_child_parent = features.getParent_Children(fileName_Contents);
    fileName_avgResponses = features.findAverageNumberOfResponses(file_parent_children);
    
    fileName_balance_global = features.findBalanceGlobal(fileName_Contents)
    
    postID_Text = features.getPostID_Text(fileName_Contents)
    fileName_balance_local = features.findBalanceLocal(fileName_Contents, file_child_parent, postID_Text)
    
    fileName_leafPosts = features.getFilename_LeafPosts(file_child_parent)
    fileName_numUnansweredQuestions = features.getNumUnansweredQuestions(fileName_leafPosts, postID_Text)
    
    
    data = {}
    for key, value in fileName_Labels.items():
        data[key] =  fileName_Contents[key];

    
    filenumber_index_mapping, index_filenumber_mapping = getFilenumberIndexMapping(data.keys())
    #print (filenumber_index_mapping);
    
    X = features.generateFeatureVectors(data, filenumber_index_mapping, averageDepth, averageNumberOfPosts, averageNumberOfWords, 
                                        fileName_avgResponses, fileName_balance_global, fileName_balance_local, fileName_numUnansweredQuestions);
    print (X);
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            print (X[i,j])
        print('\n')
    #sys.exit(0);
    
    #trainingFilenumbers, testFilenumbers = getTrainAndTestFiles(filenumber_index_mapping)
    binaryLabels = getBinaryLabels(fileName_Labels.values());
    trainingFilenumberFolds, testFilenumberFolds = getFoldsTrainAndTestFiles(X, binaryLabels, index_filenumber_mapping)
    
    numFolds = len(trainingFilenumberFolds);
    precisionPerFold = []
    recallPerFold = []
    fScorePerFold = []
    FPperFold = []
    FNperFold = []
    TPperFold = []
    TNperFold = []
    
    for trainingFilenumbers, testFilenumbers in zip(trainingFilenumberFolds, testFilenumberFolds):
        
        X_train, Y_train, X_test, Y_test = getTrainAndTestData(filenumber_index_mapping, X, fileName_Labels, trainingFilenumbers, testFilenumbers)
    
        percentPosTrain, percentNegTrain = getStatisticsBinary(Y_train)
        percentPosTest, percentNegTest = getStatisticsBinary(Y_test)
    
        print (percentPosTrain, percentNegTrain)
        print (percentPosTest, percentNegTest)
    
        Y_predict = runSVM(X_train, Y_train, X_test, Y_test);
        result = getScores(Y_test, Y_predict)
        print ('Precision','Recall','F-score','FP, FN, TP, TN') 
        print (result)
        
        precisionPerFold.append(result[0])
        recallPerFold.append(result[1])
        fScorePerFold.append(result[2])
        FPperFold.append(result[3])
        FNperFold.append(result[4])
        TPperFold.append(result[5])
        TNperFold.append(result[6])
        
    avgPrecision = np.mean(precisionPerFold)
    avgRecall = np.mean(recallPerFold)
    avgFScore = np.mean(fScorePerFold)
    avgFP = np.mean(FPperFold)
    avgFN = np.mean(FNperFold)
    avgTP = np.mean(TPperFold)
    avgTN = np.mean(TNperFold)
    
    stdPrecision = np.std(precisionPerFold)
    stdRecall = np.std(recallPerFold)
    stdFScore = np.std(fScorePerFold)
    stdFP = np.std(FPperFold)
    stdFN = np.std(FNperFold)
    stdTP = np.std(TPperFold)
    stdTN = np.std(TNperFold)
    
    print ('\n\n final')
    print ('Precision','Recall','F-score','FP, FN, TP, TN') 
    print (avgPrecision, avgRecall, avgFScore, avgFP, avgFN, avgTP, avgTN)
    print('std deviation')
    print (stdPrecision, stdRecall, stdFScore, stdFP, stdFN, stdTP, stdTN)
    
    numFolds = len(trainingFilenumberFolds);
    precisionPerFold = []
    recallPerFold = []
    fScorePerFold = []
    FPperFold = []
    FNperFold = []
    TPperFold = []
    TNperFold = []
    
    sys.exit(0);
    
    #bagofwords
    for trainingFilenumbers, testFilenumbers in zip(trainingFilenumberFolds, testFilenumberFolds):
        trainingData, trainingLabels, testData, testLabels = runBagOfWords(data, fileName_Labels, filenumber_index_mapping, index_filenumber_mapping, trainingFilenumbers, testFilenumbers);

        Y_predict = runSVM(trainingData, trainingLabels, testData, testLabels);
        result = getScores(testLabels, Y_predict)
        print (result)
        
        precisionPerFold.append(result[0])
        recallPerFold.append(result[1])
        fScorePerFold.append(result[2])
        FPperFold.append(result[3])
        FNperFold.append(result[4])
        TPperFold.append(result[5])
        TNperFold.append(result[6])
    
    avgPrecision = np.mean(precisionPerFold)
    avgRecall = np.mean(recallPerFold)
    avgFScore = np.mean(fScorePerFold)
    avgFP = np.mean(FPperFold)
    avgFN = np.mean(FNperFold)
    avgTP = np.mean(TPperFold)
    avgTN = np.mean(TNperFold)
    
    stdPrecision = np.std(precisionPerFold)
    stdRecall = np.std(recallPerFold)
    stdFScore = np.std(fScorePerFold)
    stdFP = np.std(FPperFold)
    stdFN = np.std(FNperFold)
    stdTP = np.std(TPperFold)
    stdTN = np.std(TNperFold)
    
    print (avgPrecision, avgRecall, avgFScore, avgFP, avgFN, avgTP, avgTN)
    print (stdPrecision, stdRecall, stdFScore, stdFP, stdFN, stdTP, stdTN)

main();