import email, os, nltk
import pandas as pd
import numpy as np

direc = './Data/Spam/BG/2004'
df=loadMail2df(direc)
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
                dat['sub']=emsg['Subject']
                txt=readBody(emsg)    
                dat['txt']=txt
                df=df.append(dat,ignore_index=True)
    return df

def readBody(emsg):
    replaceList=['\n']
    if emsg.is_multipart():        
        txt=''
        for payload in emsg.get_payload(): # loop over parts of e-mail
            txt=txt+ readBody(payload);                    
    else:
        txt=emsg.get_payload()
    if isinstance(txt,list):
        ''.join(txt)
        
#                # TODO filter out html and other nuisance text
#                for replace in replaceList:
#                    txt.replace(replace,' ')
    return txt

  # %% 
def genVocab(df, maxWords=1000):
    ndim=df.ndim
    word2index=[]
    for ix in range(ndim):
        tkn=[]
        lst=df.ix[:,ix].tolist()
        for txt in lst:
            if pd.notnull(txt):
                tkn.extend(nltk.word_tokenize(txt.lower())) # note text is forced to be lower case
        
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
            if pd.isnull(txt):
                ndf.iloc[ix][field]=np.nan
            else:
                txt=str(txt)
                txt=nltk.word_tokenize(txt.lower())
                nlist=list()
                for wrd in txt:
                    if wrd in word2index[field]:
                        nlist.append(word2index[field][wrd])
                    else:
                        nlist.append(len(word2index[field])+1) # unknown words get "unknown" value (=N+1)
                ndf.iloc[ix][field]=nlist
    return ndf
                    
    
        
        
                    