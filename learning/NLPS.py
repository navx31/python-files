import numpy as np 
import pandas as pd 
df=pd.read_csv('train.txt',sep=';',header=None,names=['text','emotion'])

unique_emotions=df['emotion'].unique()
emotion_numbers={}
i=0
for emo in unique_emotions:
    emotion_numbers[emo]=i
    i+=1
df['emotion']=df['emotion'].map(emotion_numbers)

df['text']=df['text'].apply(lambda x: x.lower())

import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab/english')
    nltk.data.find('corpora/stopwords')
except LookupError:
    import ssl, certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
def remove(txt):
    words=word_tokenize(txt)
    cleaned=[]
    for i in words:
        if not i in stop_words:
            cleaned.append(i)
    return ' '.join(cleaned)
df['text']=df['text'].apply(remove)
print(df.head())