# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:12:41 2021

@author: dell
"""
'''FAKE NEWS DETECTION USING LSTM'''
import pandas as pd
import pickle

df=pd.read_csv('D:/CSIT/7th sem/7TH SEM PROJECT/FAKE-NEWS-PROJECT/train[1].csv')
print(df.head())

df=df.dropna()

##get the independent features
X=df.drop('label',axis=1)

##get dependent features
y=df['label']

print(X.shape)
print(y.shape)

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

##defining vocabulary size
voc_size=5000

'''ONE HOT REPRESENTATION'''
messages=X.copy()
messages.reset_index(inplace=True)

import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

##data preprocessing
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
corpus= []
for i in range(0,len(messages)):
    print(i)
    review= re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review= review.lower()
    review= review.split()
    
    review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
    

print(corpus)

onehot_repr=[one_hot(words,voc_size)for words in corpus]
print(onehot_repr)

'''EMBEDDING REPRESENTATION'''
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

'''test'''
print(embedded_docs[0])

'''for model pickling later'''
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()

##creating model
embedding_vector_features=40
model= Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

print(X_final)
print(y_final)

print(X_final.shape,y_final.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=42)

'''MODEL TRAINING'''
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


'''PERFORMANCE METRICS AND ACCURACY'''
y_pred=(model.predict(X_test)>=0.5).astype(str).astype(int)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('predicted labels');ax.set_ylabel('true labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['real', 'fake']); ax.yaxis.set_ticklabels(['real', 'fake']);

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

'''saving model to disk'''
pickle.dump(model, open('model.pkl','wb'))

'''loading model to compare results'''
models=pickle.load(open('model.pkl','rb'))
