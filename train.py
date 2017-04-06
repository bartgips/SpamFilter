#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:47:25 2017

@author: bart
"""

import pandas as pd
import os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# loading necessary data:
modelDir=input('model directory? (default=../output)\n:')
modelfName=input('model file name? (default=mdl_RForest.pkl)\n:')

if modelDir=='':
    modelDir='../output'
if modelfName=='':
    modelfName='mdl_RForest.pkl'

if not os.path.isdir:
    os.mkdir(modelDir)

if os.path.isfile(os.path.join(modelDir,modelfName)):
    query=input('model already exists. Want to retrain? (y|N)')
    if query.lower()!='y':
        trainModel=False
    else:
        trainModel=True
else:
    trainModel=True

if trainModel:    
    dataPath=input('Please input path to stored training data (default=../output/traintest_vec.pkl)\n:')
    if dataPath=='':
        dataPath='../output/traintest_vec.pkl'
    
    if not os.path.isfile(dataPath):
        raise IOError('Data file does not exist')
        
    dictPath=input('Please input path to stored dictionaries (default=../output/dictionary.json)\n:')
    if dictPath=='':
        dictPath='../output/dictionary.json'
    
    with open(dictPath,'r') as fp:
        dictionary= json.load(fp)
    
    
    dfList=joblib.load(dataPath)
    
    if isinstance(dfList,list):
        df=pd.DataFrame()
        for dff in dfList:
            df=df.append(dff)
            
    Y=df['spamFlag'] #=first column of df
    X=df.iloc[:,1:] # drop first column (=spamFlag)
    
    print('Fitting model...')
    # training
    forest = RandomForestClassifier(400,criterion='gini',max_features = 'auto',oob_score=True, n_jobs=-1, verbose=False,)
    forest.fit(X,Y)
    
    print('Saving model to disk (' + os.path.join(modelDir,modelfName) + ')')
    joblib.dump([forest,dictionary],os.path.join(modelDir,modelfName))
else:
    print('No model was trained.')
    
    
