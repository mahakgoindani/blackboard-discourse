#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:06:50 2018

@author: csuser
"""

import pickle
import numpy as np
import sys
import math

from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import re

from nltk import ngrams
from operator import itemgetter

def remove_stopwords(l_words, lang='english'):
	l_stopwords = stopwords.words(lang)
        #l_stopwords.remove('not')
	content = [w for w in l_words if w.lower() not in l_stopwords]
	return content

def tokenize(str):
	'''Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words'''
	return 	[word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
			for t in sent_tokenize(str.replace("'", ""))]
				
#Stem all words with stemmer of type
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
	supported_stemmers = ["PorterStemmer", "WordNetLemmatizer"]
	if type is False or type not in supported_stemmers:
		return words_l
	else:
		l = []
		if type == "PorterStemmer":
			stemmer = PorterStemmer()
			for word in words_l:
				l.append(stemmer.stem(word).encode(encoding))
		if type == "WordNetLemmatizer": 
			wnl = WordNetLemmatizer()
			for word in words_l:
				l.append(wnl.lemmatize(word).encode(encoding))
		return l    
    
def preprocess_pipeline(str, lang="english", stemmer_type="PorterStemmer", return_as_list=False, do_remove_stopwords=True):
    
    l = []
    words = []
        
    sentences = tokenize(str)
    #print (sentences)
    #print (len(sentences))
    #sys.exit(0);
    for sentence in sentences:
    		if do_remove_stopwords:
    			words = remove_stopwords(sentence, lang)
    		else:
    			words = sentence
    		words = stemming(words, stemmer_type)
    		l.append(words)
    	
    if return_as_list:
        new = []
        map(new.extend, l)
        return new 
    else:
        return l
    
    
def markerstovec(text):
    f = open('markers.txt', 'r')
    markers = f.readlines()
    #print(markers)
    #print(len(markers))
    features = []
    for each in markers:
        each = each.strip()
        if(len(each) == 0):
            continue
        lis = each.split(" ,")
        res = preprocess_pipeline(" ".join(lis), lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
        elem = "_".join(res)
        features.append(elem)
    
    #print(features)
    
    size = len(features)
    vec = np.zeros(size)
    
    n = 0
    count = 0
    for n in xrange(1,6):
        grams = ngrams(text.split(), n)
        for gram in grams:       
            res = preprocess_pipeline(" ".join(gram), lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=False)
            elem = "_".join(res)
            if elem in features:
                inx = features.index(elem)
                vec[inx] += 1
                count = count+1 
    
    #print(len(vec))
    return vec

def extract_best_bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=500, freq_filter = 3):
    #bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(words, window_size=3)
    
    if(freq_filter > 0):
        bigram_finder.apply_freq_filter(freq_filter)
    
    bigrams = bigram_finder.nbest(score_fn, n)
    
    return ["_".join(bigram) for bigram in bigrams]
    

def feature_extractor(data, labels):
        
    data = " ".join(data)
    data = data.decode('ascii', 'ignore')

    #print(len(new_data))

    res = preprocess_pipeline(data, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)

    words = res;
    #print words;
    #sys.exit(0);

    word_fd = {}
     
    for word in res:
        if(word.lower() not in word_fd.keys()):
            word_fd[(word.lower())] = 1;
        else:
            word_fd[(word.lower())] += 1
    
    #best = sorted(word_fd.items(), key=lambda (w,s): s, reverse=True)[:3000]
    best = sorted(word_fd.items(), key=itemgetter(1),reverse=True)[:5000]
    
    bestwords = set([w for w, s in best])

    bigrams = extract_best_bigrams(words)
    #print(list(bestwords))
    #print(bigrams)
 
    features = list(bestwords) + bigrams
    #print features

    return features

#Converts each sentence to vector form based on features    
def prepare_any_data(data, features):
    processed_data = []
    for each in data:
        each = each.decode('ascii', 'ignore')
        #print(each)
        res = preprocess_pipeline(each, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
        #print(res)
        processed_data.append(res)
        
    vec_data = []
    for each in processed_data:
        bigrams = extract_best_bigrams(each, freq_filter=0)
        each = each + bigrams
        #print(each)
        vec = compute_feature_vector(each, features)
        vec_data.append(vec)
    vec_data = np.array(vec_data)
    return vec_data

def compute_feature_vector(arr, features):
    size = len(features)
    count = 0
    vec = np.zeros(size)
    for each in arr:
        each = each.lower()
        if each in features:
            inx = features.index(each)
            vec[inx] += 1
            count = count+1
    
    print(vec);
    #vec = vec/len(arr);    
    #print(vec);    
    
    return vec

def runBagOfWords(data, fileName_Labels, filenumber_index_mapping, index_filenumber_mapping, trainingFilenumbers, testFilenumbers):
    
    textInFile = [];
    labels = []
    for filenumber, value in data.items():
        
        label = fileName_Labels[filenumber][1];
        main_post_text = value[4][0];
        text = main_post_text.strip();
        listOfResponses = value[5];
        
        for i in range(len(listOfResponses)):
            text = text + ' ' + listOfResponses[i][0].strip();
            
        #listOfWordsInFile = text.split();
        textInFile.append(text);
        labels.append(label);
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for filenumber in trainingFilenumbers:
        index = filenumber_index_mapping[filenumber];
        X_train.append(textInFile[index])
        Y_train.append(labels[index]);
        
    for filenumber in testFilenumbers:
        index = filenumber_index_mapping[filenumber];
        X_test.append(textInFile[index])
        Y_test.append(labels[index])
    
    features = feature_extractor(X_train, Y_train)
    
    trainingData = prepare_any_data(X_train, features)
    testData = prepare_any_data(X_test, features)
    
    print (trainingData)
    print (testData)

#    shuffle_indices = np.random.permutation(np.arange(len(trainingData)))
#    
#    trainingData = trainingData[shuffle_indices]
#    
#    trainingLabels = trainingLabels[shuffle_indices]
    
    #runSVM(trainingData, trainingLabels, testData, testLabels);
    
    return trainingData, Y_train, testData, Y_test