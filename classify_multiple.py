#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:21:44 2017

@author: bart
"""
############
# Classifies all e-mail message in a directory using model located in ../output/mdl_RForest.pkl
############
from preproc import readBody
from preproc import loadMail2df, txt2vecDF
import sys
from sklearn.externals import joblib

# %%
print('Loading model in ../output/mdl_RForest.pkl')
[forest,dictionary]=joblib.load('../output/mdl_RForest.pkl')

try:
    dataPath=sys.argv[1]
except:
    dataPath=input('please provide path to e-mail file.\n:')
    
df=loadMail2df(dataPath)
dfV=txt2vecDF(df,dictionary)

labs=['Ham','Spam']
predLab, predProb=[forest.predict(dfV), forest.predict_proba(dfV)]

print(predLab)
    
