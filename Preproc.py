import email, os, nltk, re
import pandas as pd
import numpy as np

spamdirec = './Data/Spam/'
hamdirec = './Data/Ham/beck-s/'
dfSpam=loadMail2df(spamdirec)
dfHam=loadMail2df(hamdirec)




#df.to_csv('test.csv',index=False)

#df=pd.read_csv('test.csv')


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
                    dat['sub']=' '
                txt=readBody(emsg)    
                dat['txt']=txt
                df=df.append(dat,ignore_index=True)
    return df

def readBody(emsg):
    if emsg.is_multipart():        
        txt=''
        for payload in emsg.get_payload(): # loop over parts of e-mail
            txt=txt+ ''.join(readBody(payload));                    
    else:
        txt=emsg.get_payload()
    if isinstance(txt,list):
        ''.join(txt)
        
    txt=txt.lower() # make all text lower case
    # TODO maybe keep track of ratio of uppercase/lowercase as useful feature
    
    txt=re.sub('-',' ',txt) # change hyphens to spaces
    
    #filter out html and other nuisance text
    # TODO maybe find a proper way to extract text from html (parse)
    replaceList=['\n', '<head.+>','<body.+>','<title.+>','<div.+>','<\.+>','<!doctype.+>','[?!.,:;\'\"]']
    for replace in replaceList:
        txt=re.sub(replace,'',txt)
    
    txt=txt.split(' ')#nltk.word_tokenize(txt)
    for ix in range(len(txt),0,-1):
        word = txt[ix-1]
        if len(word)>45: # longest word in dictionary (see https://en.wikipedia.org/wiki/Longest_word_in_English)
            txt[ix-1]='WORD_TOO_LONG'
        elif len(word)<2: # remove single characters ('a', or floating punctuation)
            txt.pop(ix-1)
    
    return txt

  # %% 
def genVocab(df, maxWords=1000):
    ndim=df.ndim
    word2index=[]
    for ix in range(ndim):
#        tkn=[]
#        lst=df.ix[:,ix].tolist()
#        for txt in lst:
#            if pd.notnull(txt):
#                tkn.extend(nltk.word_tokenize(txt.lower())) # note text is forced to be lower case
        
        # concantenate all words
        tkn=[]
        lst=df.ix[:,ix].tolist()
        for txt in lst:
            tkn.extend(txt)
        # count word frequencies
        wordFreq= nltk.FreqDist(tkn)
        
        # extract 'maxWords' most common words
        vocab=wordFreq.most_common(maxWords)
        index2word = [x[0] for x in vocab]
        word2index.append( dict([(w,i) for i,w in enumerate(index2word)]))
        
    return word2index
# %%
def tokenizeDF(df,word2index):
    ndim=df.ndim
    ndf=df.copy()
    for field in range(ndim):
        for ix in range(len(df)):
            txt=df.iloc[ix][field]
            print([ix, field])
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
                    
    
        
        
                    