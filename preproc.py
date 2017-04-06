import os, nltk, re, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from math import *

def main():
        # %% load data and preprocess to words
    savDir= input('Please input the Output directory (default=../output)\n:')
    if savDir=='':
        savDir='../output/'
    
    if not os.path.isdir(savDir):
        os.mkdir(savDir)
    
    nWords=[input('how many words for the subject dictionary? (500)\n:')]
    nWords.append(input('how many words for the body text dictionary? (1000)\n:'))
    nWordsDefault=[500,1000] #smaller dictionay for subject    
    for n in range(len(nWords)):
        try:
            nWords[n]=int(nWords[n])
        except:
            nWords[n]=nWordsDefault[n]
    
    
    if os.path.isfile(os.path.join(savDir,'traintest_txt.pkl')):
        query=input('Files containing extracted text already exist. Redo extraction? (y|N):')
        if query.lower()=='y':
            doExtract=True
        else:
            doExtract=False
            print('loading previously extracted text')        
            #loading 
            dfTrain, dfTest=joblib.load(os.path.join(savDir,'traintest_txt.pkl'))
            
    if doExtract:        
        spamdirec = input('Please input the Spam directory (default=../Data/Spam)\n:')
        if spamdirec=='':
            spamdirec='../Data/Spam/'
        
        hamdirec = input('Please input the Spam directory (default=../Data/Ham)\n:')
        if hamdirec=='':
            hamdirec = '../Data/Ham/'                  
        print('Importing emails...')
        # load spam and ham into dataframes
        dfSpam=loadMail2df(spamdirec)
        dfHam=loadMail2df(hamdirec)
            
        #split into train and test data after scrambling
#        dfSpam=dfSpam.sample(frac=1).reset_index(drop=True)
#        dfHam=dfHam.sample(frac=1).reset_index(drop=True)
        dfTot=pd.concat([dfSpam,dfHam]).reset_index(drop=True)
        Y=np.concatenate((np.ones(len(dfSpam)), np.zeros(len(dfHam))))
        Y=pd.DataFrame(Y,index=range(len(dfTot)),columns=['spamFlag'])
        
        # add class lables to first column
        dfTot=pd.concat([Y,dfTot],axis=1)
        
        dfTrain, dfTest, Ytrain, Ytest = train_test_split(dfTot,Y,test_size=0.1)
        
        dfTrain=dfTrain.reset_index(drop=True)
        dfTest=dfTest.reset_index(drop=True)
        
        print('saving extracted text')
        # save extracted text
        joblib.dump([dfTrain, dfTest],os.path.join(savDir,'traintest_txt.pkl'))

    print('Generating dictionaries')            
    dictionary=genVocab(dfTrain.iloc[:,1:],nWords)
    
    print('saving dictionaries')
    # save dictionary for use later on (when classifying new samples after training)
    with open(os.path.join(savDir,'dictionary.json'),'w') as fp:
        json.dump(dictionary, fp)
        
    ### to load: (for now just generate, it takes almost no time)
    #with open('../Data/dictionary.json','r') as fp:
    #    dictionary= json.load(fp)
    
    print('Converting text to frequency vectors (this may take some time)...')
    dfTrainVec=pd.concat([dfTrain.iloc[:,0], txt2vecDF(dfTrain.iloc[:,1:],dictionary)],axis=1)
    dfTestVec=pd.concat([dfTest.iloc[:,0], txt2vecDF(dfTest.iloc[:,1:],dictionary)],axis=1)
    
    print('saving frequency vectors')
    joblib.dump([dfTrainVec, dfTestVec],os.path.join(savDir,'traintest_vec.pkl'))
    
#    #loading
#    dfTrainVec, dfTestVec = joblib.load('../Data/preproc/traintest_vec.pkl')
    print('Done!')


# %%
def loadMail2df(direc):
    # function to load single e-mail or recursively run through a directory
    # returns a dataframe that includes the subject and body text. The text 
    # is split up into a list where every element is a word.
    import email, os, nltk, re
    import pandas as pd
    from preproc import readBody
    
    cols=['sub','txt']
    df=pd.DataFrame(columns=cols)
    
    if os.path.isdir(direc):
        for (dirpath, dirnames, filenames) in os.walk(direc):
            fpaths=[]
            for fname in filenames: # loop over files
                fpaths.append(os.path.join(dirpath, fname))
    elif os.path.isfile(direc):
        fpaths=[direc]
    else:
        raise IOError('Data file does not exist')
            
    for fpath in fpaths:    
        fid=open(fpath)
        try:
            emsg=email.message_from_file(fid)
            readCorrectly=True
        except:
            readCorrectly=False
            raise ValueError('waaa')
        fid.close()
        if readCorrectly:
            dat={}
            subDum=emsg['Subject']
            if not subDum is None:
                subDum=subDum.lower() #make text lower case
                subDum=re.sub('-',' ',subDum) #change hyphens to spaces
                
                replaceList= ['\n'] #keep punctuation in for nltk.word_tokenize
                for replace in replaceList:
                    subDum=re.sub(replace,'',subDum)
                subDum=nltk.word_tokenize(subDum)
                dat['sub']=subDum
            else:
                dat['sub']=[' ']
            txt=readBody(emsg)    
            dat['txt']=txt
            df=df.append(dat,ignore_index=True)
    return df

def readBody(emsg):
    # helper function
    # returns body text from email object
    import re, nltk
    
    if emsg.is_multipart():        
        txt=''
        for payload in emsg.get_payload(): # loop over parts of e-mail
            txt=txt+ ' '.join(readBody(payload));                    
    else:
        txt=emsg.get_payload()
    if isinstance(txt,list):
        ''.join(txt)
        
    txt=txt.lower() # make all text lower case
    # TODO maybe keep track of ratio of uppercase/lowercase as useful feature
    
    txt=re.sub('-',' ',txt) # change hyphens to spaces
    txt=re.sub('\n',' ',txt) # change newlines to spaces
#    txt=re.sub('[?!.,:;\'\"]',' ',txt) # change punctuation to spaces
    
    #filter out html and other nuisance text
    # TODO maybe find a proper way to extract text from html (parse)
    replaceList=['<html.+>', '<head.+>','<body.+>','<title.+>','<div.+>','<\.+>','<!doctype.+>']
    for replace in replaceList:
        txt=re.sub(replace,'',txt)
    
    txt=nltk.word_tokenize(txt)
    for ix in range(len(txt),0,-1):
        word = txt[ix-1]
        if len(word)>45: # longest word in dictionary (see https://en.wikipedia.org/wiki/Longest_word_in_English)
#            raise ValueError('something')
            txt[ix-1]='WORD_TOO_LONG'
        elif len(word)<2: # remove single characters ('a', or floating punctuation)
            txt.pop(ix-1)
    
    return txt


def genVocab(df, maxWords=1000):
    # function that generates dictionary/vocabulary of size 'maxWords' for 
    # every column in df.
    import nltk
    
    ndim=df.ndim
    dictionary=[]
    if isinstance(maxWords,int):
        maxWords=[maxWords] * ndim
    for ix in range(ndim):
        N=maxWords[ix]
        tkn=[]
        lst=df.ix[:,ix].tolist()
        for txt in lst:
            tkn.extend(txt)
        # count word frequencies
        wordFreq= nltk.FreqDist(tkn)
        
        # extract 'maxWords' most common words
        vocab=wordFreq.most_common(N)
        index2word = [x[0] for x in vocab]
        dictionary.append( dict([(w,i) for i,w in enumerate(index2word)]))
        
    return dictionary

                    
def txt2vecDF(df,dictionary):
    # converts parsed texts into word frequency vectors using the vocabulary in
    # 'dictionary'.
    import pandas as pd
    
    # convert extracted text to frequency vectors using dictionary
    ndim=df.ndim
    ndf=pd.DataFrame(None)
    loopfields=range(ndim)
    collabs=df.columns
    for field in loopfields:
        cols=[collabs[field] +' : ' + s for s in list(dictionary[field].keys()) + ['_unknown','_txtLen']]
        dumdf=pd.DataFrame(columns=cols)
        for txt in df.iloc[:,field]: # consider backwards looping over indices to pre-allocate memory for full DF
            if len(txt)<1: # fill with zeros
                dumdf=dumdf.append(pd.DataFrame(0, index = [0],columns=cols))
            else:
                Nbin=len(dictionary[field])
                nlist=np.zeros((Nbin+2,1))
                for wrd in txt:
                    if wrd in dictionary[field]:
                        nlist[dictionary[field][wrd]]=nlist[dictionary[field][wrd]]+1
                    else:
                        nlist[Nbin]=nlist[Nbin]+1 # unknown words get "unknown" value (=N+1)
                nlist=nlist/len(txt) # normalize to give frequencies
                nlist[Nbin+1]=len(txt) # final feature is _txtLen (i.e. number of words)
                dumdf=dumdf.append(pd.DataFrame(nlist.transpose(),index=[0],columns=cols))
        ndf=pd.concat([ndf,dumdf],axis=1)
    ndf=ndf.reset_index(drop=True)
    return ndf

if __name__ == '__main__':
    main()
                    