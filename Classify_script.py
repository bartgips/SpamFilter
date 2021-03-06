#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:19:19 2017

@author: bart
"""
# script to generate some of the figures used in the presentation

#import json
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# %% read the data
savDir= input('Please input the Output directory (default=../output)\n:')
if savDir=='':
    savDir='../output/'
    
dfTrainVec, dfTestVec =joblib.load(os.path.join(savDir,'traintest_vec.pkl'))

Y = dfTrainVec['spamFlag']
Y_test = dfTestVec['spamFlag']   
X = dfTrainVec.iloc[:,1:]
X_test = dfTestVec.iloc[:,1:]


# %% random forest classification
figDir='../figs/'
labs=X.columns
n_est=[10,20,30,40,50,60,70,80,100,200,500,1000]
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
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), labs[indices], rotation='vertical')
    plt.xlim([-.5, 10.5])
    plt.tight_layout
    cnt+=1
plt.show(1)
# %%

fig4 = plt.figure(4)
plt.subplot(1,2,1)
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labs[indices], rotation='vertical')
plt.xlim([-0.5, 15.5])
plt.tight_layout
plt.ylabel('feature importance')
plt.xlabel('feature')
plt.subplot(1,2,2)
plt.plot(range(X.shape[1]), importances[indices], color="r")
plt.xlim([0, 300])
plt.ylim([0, 0.05])
plt.tight_layout
plt.show(fig4)
fig4.set_size_inches(11,8)
plt.xlabel('feature')
fig4.tight_layout()
#plt.suptitle("Feature importances: " + str(n_trees) + ' trees')

plt.savefig(os.path.join(figDir,'Feat_importance.pdf'))
    

fig2 = plt.figure(2)
plt.clf()
plt.title('out of bag performance')
plt.plot(n_est,perf)
plt.xlabel('Forest size')
plt.ylabel('performance')
plt.show(2)
fig2.set_size_inches(10,5)
plt.savefig(os.path.join(figDir,'OOBperf_v_nTrees.pdf'))


#joblib.dump(forest,'../models/forest.pkl')

# to load
#forest=joblib.load('./models/forest.pkl')
# %% X-validation
#forest = RandomForestClassifier(n_estimators=800,criterion='gini',max_features = 'auto',oob_score=True, n_jobs=-1, verbose=False)
predict=cross_val_predict(forest,X,Y,cv=10)

Y_test_pred=forest.predict(X_test)
cm=confusion_matrix(Y,predict).astype(float)
cmTest=confusion_matrix(Y_test,Y_test_pred).astype(float)
for n in range(len(cm[0,:])):
    cm[:,n]=cm[:,n]/np.sum(cm[:,n])
    cmTest[:,n]=cmTest[:,n]/np.sum(cmTest[:,n])

#%%
fig3=plt.figure(3)
plt.clf()
titlabs=['10-fold x-val','test data']
performance=[1.0-np.mean(abs(predict-Y)), 1.0-np.mean(abs(Y_test-Y_test_pred))]
for CM,spi,perf in zip([cm,cmTest],range(2),performance):
    plt.subplot(1,2,spi+1)
    plt.imshow(CM)
    plt.title('confusion matrix of ' + titlabs[spi] + '\nperformance = ' + str(perf))
    plt.xticks([0, 1], ['Ham','Spam'])
    plt.yticks([0, 1], ['Ham','Spam'])
    plt.xlabel('predicted')
    plt.ylabel('true label')
    plt.set_cmap('magma')
    plt.clim([0,1])
    plt.colorbar()
    plt.hold(True)
    for xx in range(2):
        for yy in range(2):
            plt.text(yy,xx,'{:.2f}%'.format(CM[xx,yy]*100),horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white'))
            
            
fig3.set_size_inches(10, 5)

fig3.savefig(os.path.join(figDir,'ConfusionMat_randFor.pdf'))
# %% naive bayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
    
alist=np.linspace(-2,3,21)
accMNB=[]
accBNB=[]
for alph in alist:
    MNBmodel=MultinomialNB(alpha=10**alph)
    MNBpredict=cross_val_predict(MNBmodel,X,Y,n_jobs=-1,cv=5)
    accMNB.append(1-np.mean(abs(MNBpredict-Y)))
    
    BNBmodel=BernoulliNB(alpha=10**alph)
    BNBpredict=cross_val_predict(BNBmodel,X,Y,n_jobs=-1,cv=5)
    accBNB.append(1-np.mean(abs(BNBpredict-Y)))

GNBmodel=GaussianNB()
GNBpredict=cross_val_predict(GNBmodel,X,Y,n_jobs=-1,cv=5)
accGNB=1-np.mean(abs(GNBpredict-Y))

fig10=plt.figure(10)
fig10.clf
plt.plot(alist,accMNB,alist,accBNB,alist,[accGNB]*len(alist))
plt.ylabel('performance')
plt.xlabel('smoothing prior (alpha) (log10)')
plt.legend(['Multinomial','Bernoulli','Gaussian'])

fig10.savefig(os.path.join(figDir,'NaiveBayes2.pdf'))


# %% svm
from sklearn import svm

supVec=svm.SVC()
supVec.fit(X,Y)



confusion_matrix(Y_test,supVec.predict(X_test))










