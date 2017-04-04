import email, os, nltk, re, json
import pandas as pd
import numpy as np

spamdirec = './Data/Spam/'
hamdirec = './Data/Ham/'

# %% load data and preprocess to words
dfSpam=loadMail2df(spamdirec)
dfHam=loadMail2df(hamdirec)

#dfSpam.to_csv('./Data/Spam_txt.csv',index=False)
dfSpam.to_json('./Data/Spam_txt_more.json')
dfHam.to_json('./Data/Ham_txt_more.json')


## to load:
#dfSpam=pd.read_json('./Data/Spam_txt.json')
#dfHam=pd.read_json('./Data/Ham_txt.json')

# %%
nWords=[500,1000] #smaller dictionay for subject
dictionary=genVocab(pd.concat([dfSpam,dfHam]),nWords)

with open('./Data/dictionary_more.json','w') as fp:
    json.dump(dictionary, fp)
    
## to load:
#with open('./Data/dictionary.json','r') as fp:
#    dictionary= json.load(fp)


# %%

import time
t= time.time()
dfSpamV=txt2vecDF(dfSpam,dictionary)
elapsed1 = time.time() - t


dfHamV=txt2vecDF(dfHam,dictionary)
elapsed2 = time.time() - t - elapsed1



# save
dfSpamV.to_json('./Data/Spam_Vec_more.json')
dfHamV.to_json('./Data/Ham_Vec_more.json')





# %%
def loadMail2df(direc):
    #direc = './Data/Ham/beck-s/'
    cols=['sub','txt']
    df=pd.DataFrame(columns=cols)
    for (dirpath, dirnames, filenames) in os.walk(direc):
        for fname in filenames: # loop over files
            fpath=os.path.join(dirpath, fname)
            fid=open(fpath)
            try:
                emsg=email.message_from_file(fid)
                readCorrectly=True
            except:
                readCorrectly=False
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
    txt=re.sub('[?!.,:;\'\"]',' ',txt) # change punctuation to spaces
    
    #filter out html and other nuisance text
    # TODO maybe find a proper way to extract text from html (parse)
    replaceList=['<html.+>', '<head.+>','<body.+>','<title.+>','<div.+>','<\.+>','<!doctype.+>']
    for replace in replaceList:
        txt=re.sub(replace,'',txt)
    
    txt=txt.split(' ')#nltk.word_tokenize(txt)
    for ix in range(len(txt),0,-1):
        word = txt[ix-1]
        if len(word)>45: # longest word in dictionary (see https://en.wikipedia.org/wiki/Longest_word_in_English)
#            raise ValueError('something')
            txt[ix-1]='WORD_TOO_LONG'
        elif len(word)<2: # remove single characters ('a', or floating punctuation)
            txt.pop(ix-1)
    
    return txt


def genVocab(df, maxWords=1000):
    ndim=df.ndim
    word2index=[]
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
        word2index.append( dict([(w,i) for i,w in enumerate(index2word)]))
        
    return word2index

def txt2vec(df,word2index):
    ndim=df.ndim
    ndf=df.copy()
    for field in range(ndim):
        for ix in range(len(df)):
            txt=df.iloc[ix][field]
#            print([ix, field])
            if all(pd.isnull(txt)):
                ndf.iloc[ix][field]=np.nan
            else:
                N=len(word2index[field])
                nlist=np.zeros((N+2,1)) # one entry for all words in dictionary, one for unknown words, one for total number of words
                for wrd in txt:
                    if wrd in word2index[field]:
                        nlist[word2index[field][wrd]]=nlist[word2index[field][wrd]]+1
                    else:
                        nlist[N]=nlist[N]+1 # unknown words get "unknown" value (=N+1)
                nlist=nlist/len(txt)
                nlist[N+1]=len(txt)
                ndf.iloc[ix][field]=nlist
    return ndf
                    
def txt2vecDF(df,word2index):
    ndim=df.ndim
    ndf=pd.DataFrame(None)
    for field in range(ndim):
        cols=[str(field).zfill(3) +'_' + s for s in list(word2index[field].keys()) + ['_unknown','_txtLen']]
        dumdf=pd.DataFrame(columns=cols)
        for txt in df.iloc[:,field]: # consider backwards looping over indices to pre-allocate memory for full DF
            if len(txt)<1: # fill with zeros
                dumdf=dumdf.append(pd.DataFrame(0, index = [0],columns=cols))
            else:
                Nbin=len(word2index[field])
                nlist=np.zeros((Nbin+2,1))
                for wrd in txt:
                    if wrd in word2index[field]:
                        nlist[word2index[field][wrd]]=nlist[word2index[field][wrd]]+1
                    else:
                        nlist[Nbin]=nlist[Nbin]+1 # unknown words get "unknown" value (=N+1)
                nlist=nlist/len(txt)
                nlist[Nbin+1]=len(txt)
                dumdf=dumdf.append(pd.DataFrame(nlist.transpose(),index=[0],columns=cols))
        ndf=pd.concat([ndf,dumdf],axis=1)
    ndf=ndf.reset_index(drop=True)
    return ndf
    

def txt2vecDF2(df,word2index):
    ndim=df.ndim
    ndf=pd.DataFrame(None)
    for field in range(ndim):
        cols=[str(field).zfill(3) +'_' + s for s in list(word2index[field].keys()) + ['_unknown','_txtLen']]
        dumdf=pd.DataFrame(None,index = range(len(df)), columns=cols)
        for ix in range(len(df),0,-1):
            txt=df.iloc[ix-1][field]
            if len(txt)<1: # fill with zeros
                dumdf.loc[ix-1,:]=0
            else:
                Nbin=len(word2index[field])
                nlist=np.zeros((Nbin+2,1))
                for wrd in txt:
                    if wrd in word2index[field]:
                        nlist[word2index[field][wrd]]=nlist[word2index[field][wrd]]+1
                    else:
                        nlist[Nbin]=nlist[Nbin]+1 # unknown words get "unknown" value (=N+1)
                nlist=nlist/len(txt)
                nlist[Nbin+1]=len(txt)
                dumdf.loc[ix-1,:]=nlist.transpose()
        ndf=pd.concat([ndf,dumdf],axis=1)
#    ndf=ndf.reset_index()
    return ndf
        
        
                    