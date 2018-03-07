#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:13:57 2018

@author: csuser
"""

import numpy as np
import re
import sys


def findAverageDepth(fileName_Contents):
    
    depthsOfPost = [];
    for key, value in fileName_Contents.items():
        depth = value[6];
        #print (depth);
        depthsOfPost.append(depth);
    averageDepth = np.mean(depthsOfPost);
    print (averageDepth);
    return averageDepth;
            
def findAverageNumberOfPosts(fileName_Contents):
    
    numberOfPosts = [];
    for key, value in fileName_Contents.items():
        totalNumberOfPosts = value[7];
        #print (totalNumberOfPosts);
        numberOfPosts.append(totalNumberOfPosts);
    averageNumberOfPosts = np.mean(numberOfPosts);
    print (averageNumberOfPosts);
    return averageNumberOfPosts;

def findAverageNumberOfWords(fileName_Contents):
    
    numberOfWords = [];
    for key, value in fileName_Contents.items():
        totalNumberOfWords = value[8];
        #print (totalNumberOfWords);
        numberOfWords.append(totalNumberOfWords);
    averageNumberOfWords = np.mean(numberOfWords);
    print (averageNumberOfWords);
    return averageNumberOfWords;

#(Forum, ForumID, Thread, ThreadID, main_post, listOfResponses, maxDepth, totalNumberOfPosts, totalNumberOfWords);   
#response = (Text, PostID, Time, UserID, ThreadID, Hits, ParentID, Subject, depth)
def getParent_Children(fileName_Contents):
    
    file_parent_children = {}
    file_child_parent = {}
    for key, value in fileName_Contents.items():
        main_post = value[4];
        main_post_ID = main_post[1];
        main_post_Parent_ID = main_post[6];
        #print (main_post)
        #print (key)
        #print (main_post_Parent_ID)
        #sys.exit(0);
        file_child_parent[key, main_post_ID] = main_post_Parent_ID
        if((key, main_post_Parent_ID) in file_parent_children.keys()):
            children = file_parent_children[(key, main_post_Parent_ID)];
            children.append(main_post_ID);
            file_parent_children[(key, main_post_Parent_ID)] = children
        else:
            file_parent_children[(key, main_post_Parent_ID)]  = [main_post_ID]
         
        listOfResponses = value[5];
        for response in listOfResponses:
            postID = response[1];
            parentID = response[6];
            file_child_parent[(key, postID)] = parentID
            if((key, parentID) in file_parent_children.keys()):
                children = file_parent_children[(key, parentID)];
                children.append(postID);
                file_parent_children[(key, parentID)] = children
            else:
                file_parent_children[(key, parentID)]  = [postID]
    return file_parent_children, file_child_parent

def findAverageNumberOfResponses(file_parent_children):
    
    fileName_numResponses = {}
    for key, value in file_parent_children.items():
        filename = key[0];
        if(filename in fileName_numResponses.keys()):
            numResponses = fileName_numResponses[filename];
            numResponses.append(len(value))
            fileName_numResponses[filename] = numResponses;
        else:
            fileName_numResponses[filename] = [len(value)]
    
    fileName_avgResponses = {}
    for key, value in fileName_numResponses.items():
        avg = np.mean(value)
        fileName_avgResponses[key] = avg;
        
    return fileName_avgResponses;

def findBalanceGlobal(fileName_Contents):
    
    fileName_balance = {};
    for key, value in fileName_Contents.items():
        main_post = value[4];
        main_text = main_post[0];
        main_post_len = len(main_text.split());
        listOfResponses = value[5];
        ratios = [];
        for response in listOfResponses:
            responseText = response[0];
            if(main_post_len!=0):
                ratio = len(responseText.split()) * 1.00/main_post_len;
            else:
                ratio = 0.0;
            ratios.append(ratio);
        avgRatio = np.mean(ratios);
        fileName_balance[key] = avgRatio;
    return fileName_balance;
        
def findBalanceLocal(fileName_Contents, file_child_parent, postID_Text):
    
    fileName_balances = {};
    
    for key, value in file_child_parent.items():
        filename = key[0];
        childID = key[1];
        parentID = value;
        childPostText = postID_Text[(filename, childID)]
        parentPostText = postID_Text[(filename, parentID)]
        if(len(parentPostText.split())!=0):
            ratio = len(childPostText.split()) * 1.00/len(parentPostText.split())
        else:
            ratio = 0.0;
        if(filename in fileName_balances.keys()):
            temp = fileName_balances[filename];
            temp.append(ratio);
            fileName_balances[filename] = temp;
        else:
            fileName_balances[filename] = [ratio]
            
    fileName_avgBalance = {};
    for key, value in fileName_balances.items():
        fileName_avgBalance[key] = np.mean(value);
        
    return fileName_avgBalance;
        

def getPostID_Text(fileName_Contents):
    
    postID_Text = {}
    for key, value in fileName_Contents.items():
        main_post = value[4];
        mainText = main_post[0];
        postID = main_post[1];
        postID_Text[(key, postID)] = mainText
        postID_Text[(key, '-1')] = mainText
        
        listOfResponses = value[5];
        for response in listOfResponses:
            responseText = response[0];
            postID = response[1];
            postID_Text[(key, postID)] = responseText
    
    return postID_Text

def getNumCitations(string):
    
    author = "(?:[A-Z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    year_num = "(?:19|20)[0-9][0-9]"
    page_num = "(?:, p.? [0-9]+)?"  # Always optional
    year = "(?:, *"+year_num+page_num+"| *\("+year_num+page_num+"\))"
    regex = "(?" + author + additional+"*" + year + ")?"

    try:
        matches = re.findall(r'^\*' + regex, string)    
    except:
        matches = [];
    
    print (matches)
    
    return len(matches)

def getNumReferencedURLS(string):
    
    #string.replace('*','')
    #urls = re.findall(r'^\*'+'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    #urls = [m.start() for m in re.finditer('http', string)]
    #print (urls)
    
    try:
        urls = re.findall(r'^\*'+'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    except:
        urls = [];
    
    return len(urls)
    
def getNumResources(string):
        
    numLinks = getNumReferencedURLS(string)
    numCitations = getNumCitations(string)
    
    return numLinks + numCitations

def getFilename_LeafPosts(file_child_parent):
    
    fileName_allPostIDs = {}
    for key in file_child_parent.keys():
        filename = key[0];
        postID = key[1];
        if(filename in fileName_allPostIDs.keys()):
            postIDs = fileName_allPostIDs[filename]
            postIDs.append(postID);
            fileName_allPostIDs[filename] = postIDs;
        else:
            fileName_allPostIDs[filename] = [postID];
            
    fileName_allParentIDs = {}
    for key in file_child_parent.keys():
        filename = key[0];
        parentID = file_child_parent[key]
        if(filename in fileName_allParentIDs.keys()):
            parentIDs = fileName_allParentIDs[filename]
            parentIDs.append(parentID);
            fileName_allParentIDs[filename] = parentIDs;
        else:
            fileName_allParentIDs[filename] = [parentID];
            
    fileName_leafPosts = {}
    for key in fileName_allPostIDs.keys():
        leafPosts = [];
        childIDs = fileName_allPostIDs[key];
        parentIDs = fileName_allParentIDs[key];
        for childID in childIDs:
            if(childID not in parentIDs):
                leafPosts.append(childID);
        fileName_leafPosts[key] = leafPosts;
    
    return fileName_leafPosts

def getNumUnansweredQuestions(fileName_leafPosts, postID_Text):
    
    fileName_numUnansweredQuestions = {}
    for key, value in fileName_leafPosts.items():
        totalNumQuestions = 0;
        for postID in value:
            postText = postID_Text[(key, postID)];
            numQuestionsInPost = postText.count('?');
            totalNumQuestions = totalNumQuestions + numQuestionsInPost
        fileName_numUnansweredQuestions[key] = totalNumQuestions
    
    return fileName_numUnansweredQuestions
            
def generateFeatureVectors(fileName_Contents, filenumber_index_mapping, averageDepth, averageNumberOfPosts, averageNumberOfWords, 
                           fileName_avgResponses, fileName_balance_global, fileName_balance_local, fileName_numUnansweredQuestions):
    
    X = np.zeros((len(fileName_Contents.keys()), 7), dtype = float)
    
    for key, value in fileName_Contents.items():
        
        index = filenumber_index_mapping[key];
        depth = value[6];
        totalNumberOfPosts = value[7];
        totalNumberOfWords = value[8];
        
         
        if(depth < averageDepth):
            X[index, 0] = -1;
        else:
            X[index, 0] = 1;
            
        if(totalNumberOfPosts < averageNumberOfPosts):
            X[index, 1] = -1;
        else:
            X[index, 1] = 1;   
             
        if(totalNumberOfWords < averageNumberOfWords):
            X[index, 2] = -1;
        else:
            X[index, 2] = 1;
        
        X[index, 3] = fileName_avgResponses[key];
        X[index, 4] = fileName_balance_global[key];
        X[index, 5] = fileName_balance_local[key];
        X[index, 6] = fileName_numUnansweredQuestions[key];
        
        #X[index, 7] = value[9]; #totalNumberOfResources
        
        
    return X;