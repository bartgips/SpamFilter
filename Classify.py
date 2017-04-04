#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:19:19 2017

@author: bart
"""
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


# %% read the data
dfSpam=pd.read_json('./Data/Spam_Vec.json')
dfHam=pd.read_json('./Data/Ham_Vec.json')


X=pd.concat([dfSpam,dfHam])
Y=np.concatenate((np.ones(len(dfSpam)), np.zeros(len(dfHam))))
# %% random forest classification

labs=X.columns
n_est=[10,20,30,40,50,60,70,80,100,200,400]
fig1=plt.figure(1)
cnt=0
perf=np.zeros(len(n_est))
for n_trees in n_est:
    print('classifying using ' + str(n_trees) + ' trees')
    forest = RandomForestClassifier(n_estimators=n_trees,criterion='gini',max_features = 'auto',oob_score=True, n_jobs=-1, verbose=False)
    forest.fit(X,Y)
    
    perf[cnt]=forest.oob_score_
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices= np.argsort(importances)[::-1]
    
##   Print the feature ranking
#    print("Feature ranking:")    
#    for f in range(X.shape[1]):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.subplot(3,4,cnt+1)
    plt.title("Feature importances: " + str(n_trees) + ' trees')
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), labs[indices], rotation='vertical')
    plt.xlim([-1, 10])
    cnt+=1
plt.show(1)
    
fig2=plt.figure(2)
plt.title('out of bag performance')
plt.plot(n_est,perf)
plt.show(2)

# %% X-validation
forest = RandomForestClassifier(n_estimators=100,criterion='gini',max_features = 'auto',oob_score=True, n_jobs=-1, verbose=False)
predict=cross_val_predict(forest,X,Y,cv=10)

fig3=plt.figure(3)
cm=confusion_matrix(Y,predict).astype(float)
for n in range(len(cm[0,:])):
    cm[:,n]=cm[:,n]/np.sum(cm[:,n])
plt.imshow(cm)
plt.title('confusion matrix')
plt.colorbar()




























