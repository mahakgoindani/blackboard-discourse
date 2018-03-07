#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:00:36 2018

@author: csuser
"""

import pickle
import os
import numpy as np
import sys
import math
from sklearn.model_selection import StratifiedKFold

import features


def loadObject(file_Name):
    fileObject = open(file_Name,'r')  
    data = pickle.load(fileObject)  
    return data;
    
def storeObject(data, file_Name):
    fileObject = open(file_Name,'wb') 
    pickle.dump(data, fileObject)
    fileObject.close()
    

def getFilenumberIndexMapping(values):
    
    filenumber_index_mapping = {}
    index_filenumber_mapping = {}
    count = 0;
    for value in values:
        filenumber_index_mapping[value] = count;
        index_filenumber_mapping[count] = value;
        count = count + 1;
        
    return filenumber_index_mapping, index_filenumber_mapping;

def getBinaryLabels(values):
    
    binaryLabels = []
    for value in values:
        binaryLabels.append(value[1])
    
    return binaryLabels

def getData(directory):
    fileName_Contents = {} # key: file number, value: file contents as a tuple of elements
    
    for filename in os.listdir(directory):
        filenumber = filename.split('.txt')[0];
        filenumber = filenumber.strip();
        #filenumber = int(filenumber);
        filename = directory + filename;
        #print (filename)
        with open(filename, 'r') as f:
            count = 0;
            maxDepth = -1;
            totalNumberOfPosts = 0;
            totalNumberOfWords = 0;
            totalNumberOfResources = 0;
            listOfResponses = []
            main_post = ();
            for line in f:
                line = line.strip()
                print (line);
                print ('end')
                count = count + 1;
                
                if(count==1):
                    continue;
                    
                elif(count==2):
                    temp = line.split('\t')
                    Forum = temp[0].split('Forum:')[1];
                    ForumID = temp[1].split('ID:')[1];
                    
                    print(Forum)
                    print(ForumID);
                    
                elif(count==3):
                    temp = line.split('\t')
                    print(temp);
                    Thread = temp[0].split('Thread:')[1];
                    ThreadID = temp[1].split('ID:')[1];
                   
                    print(Thread)
                    print(ThreadID);
                    
                elif(count==4):
                    temp = line.split('\t')
                    print (temp);
                    PostID = temp[0].split('PostID:')[1];
                    Time = temp[1].split('Time:')[1];
                    UserID = temp[2].split('UserID:')[1];
                    ThreadID = temp[3].split('ThreadID:')[1];
                    Hits = temp[4].split('Hits:')[1];
                    #ParentID = temp[5].split('ParentID:')[1];
#                    if(ParentID==''):
#                        ParentID = '-1'
                    ParentID = '-1'
                    Subject = temp[6].split('Subject:')[1];
                    depth = 0;
                    maxDepth = 0;
                    
                    print (depth)
                    print (PostID)
                    print (Time)
                    print (UserID)
                    print (ThreadID)
                    print (Hits)
                    print (ParentID)
                    print (Subject)
                    
                    
                elif(count==5):
                    Text = line.split('Text:')[1];
                    print (Text);
                    totalNumberOfWords = totalNumberOfWords + len(Text.split())
                    main_post = (Text, PostID, Time, UserID, ThreadID, Hits, ParentID, Subject, depth)
                    totalNumberOfPosts = totalNumberOfPosts + 1;
                    totalNumberOfResources = totalNumberOfResources + features.getNumResources(Text);
                    
                elif(line.startswith('##1')):
                    index = line.find('PostID:');
                    if(index!=-1):
                        depths = line[:index-1];
                        depths = depths.strip();
                        print (depths);
                        temp = depths.split();
                        depth = temp[len(temp)-1];
                        print depth;
                        depth = depth.split('##')[1];
                        depth = int (depth)
                        print depth;
                        if(depth>maxDepth):
                            maxDepth = depth;
                        
                        temp = line[index:];
                        temp = temp.strip();
                        temp = temp.split('\t')
                        print (temp);
                        PostID = temp[0].split('PostID:')[1];
                        Time = temp[1].split('Time:')[1];
                        UserID = temp[2].split('UserID:')[1];
                        ThreadID = temp[3].split('ThreadID:')[1];
                        Hits = temp[4].split('Hits:')[1];
                        ParentID = temp[5].split('ParentID:')[1];
                        if(ParentID==''):
                            ParentID = '-1'
                        Subject = temp[6].split('Subject:')[1];
                    
                    index = line.find('Text:');
                    if(index!=-1):
                        temp = line[index:];
                        Text = line.split('Text:')[1];
                        Text = Text.strip();
                        print (Text);
                        totalNumberOfWords = totalNumberOfWords + len(Text.split())
                        response = (Text, PostID, Time, UserID, ThreadID, Hits, ParentID, Subject, depth)        
                        listOfResponses.append(response);
                        totalNumberOfPosts = totalNumberOfPosts + 1;
                        totalNumberOfResources = totalNumberOfResources + features.getNumResources(Text);
                    
            fileName_Contents[filenumber] = (Forum, ForumID, Thread, ThreadID, main_post, listOfResponses, maxDepth, totalNumberOfPosts, totalNumberOfWords, totalNumberOfResources);   
            
    
    #print('data')
    #print (fileName_Contents['100005'])      
    return fileName_Contents

def getLabels(filename):
    
    fileName_Labels = {};
    count = 0;
    with open(filename, 'r') as f:
        for line in f:
            count = count + 1;
            if(count==1):
                continue;
                
            print (line);
            row = line.split(',');
            print (row)
            filenumber = row[0].split('.txt')[0].strip();
            print (filenumber)
            
            if((row[5]!='') and (row[5]!='0') and (row[5]==row[6])):
                binaryLabel = row[9];
                label = row[8];
                fileName_Labels[filenumber] = (label, binaryLabel)
                
    return fileName_Labels

def getTrainAndTestDataSVM1(filenumber_index_mapping, X, fileName_Labels):
    
    Y = [''  for x in range((len(fileName_Labels.keys())))]
    for key, value in fileName_Labels.items():
        index = filenumber_index_mapping[key];
        Y[index] = value[1];
    
#    print X.shape;
#    print len(Y);
    dataSize = len(Y)
    trainingDataSize = int (math.ceil(0.8 * dataSize))
    testDataSize = dataSize - trainingDataSize;
    print (trainingDataSize);
    print (testDataSize)
    
    X_train = X[:trainingDataSize,:]
    X_test = X[trainingDataSize:,:]
    
    Y_train = Y[:trainingDataSize]
    Y_test = Y[trainingDataSize:]
    
    return X_train, Y_train, X_test, Y_test


def getTrainAndTestFiles(filenumber_index_mapping):
    
    dataSize = len(filenumber_index_mapping.keys())
    trainingDataSize = int (math.ceil(0.8 * dataSize))
    testDataSize = dataSize - trainingDataSize;
    print (trainingDataSize);
    print (testDataSize)
    
    totalFilenumbers = filenumber_index_mapping.keys();
    trainingFilenumbers = totalFilenumbers[:trainingDataSize];
    testFilenumbers = totalFilenumbers[trainingDataSize+1:];
    
    return trainingFilenumbers, testFilenumbers

def getFoldsTrainAndTestFiles(data, labels, index_filenumber_mapping):
    
    print (labels)
    skf = StratifiedKFold(n_splits=5)
    
    trainingFilenumberFolds = []
    testFilenumberFolds = []
    
    for train_indices, test_indices in skf.split(data, labels):
        print train_indices, test_indices
        
        listOfFilenumbers = []
        for index in train_indices:
            filenumber = index_filenumber_mapping[index];
            listOfFilenumbers.append(filenumber);
        trainingFilenumberFolds.append(listOfFilenumbers)
        
        listOfFilenumbers = []
        for index in test_indices:
            filenumber = index_filenumber_mapping[index];
            listOfFilenumbers.append(filenumber);
        testFilenumberFolds.append(listOfFilenumbers)
        
    return trainingFilenumberFolds, testFilenumberFolds
            
    
def getTrainAndTestData(filenumber_index_mapping, X, fileName_Labels, trainingFilenumbers, testFilenumbers):
    
    trainingDataSize = len(trainingFilenumbers)
    testDataSize = len(testFilenumbers)
    numFeatures = X.shape[1];
    
    X_train = np.zeros((trainingDataSize, numFeatures), dtype = float)
    X_test = np.zeros((testDataSize, numFeatures), dtype = float)
    
    Y_train = [''  for x in range(trainingDataSize)]
    Y_test = [''  for x in range(testDataSize)]
    
    count = 0;
    for filenumber in trainingFilenumbers:
        index = filenumber_index_mapping[filenumber];
        X_train[count,:] = X[index,:]
        value = fileName_Labels[filenumber];
        Y_train[count] = value[1];
        count = count + 1;
        
    count = 0;
    for filenumber in testFilenumbers:
        index = filenumber_index_mapping[filenumber];
        X_test[count,:] = X[index,:]
        value = fileName_Labels[filenumber];
        Y_test[count] = value[1];
        count = count + 1;
                    
    return X_train, Y_train, X_test, Y_test


