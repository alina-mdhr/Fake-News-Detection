# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:25:38 2021

@author: dell
"""

import numpy as np
from flask import Flask,request,render_template
from flask_cors import CORS
import os
import joblib
import pickle
import flask
import newspaper
from newspaper import Article
import urllib
import nltk

nltk.download('punkt')

#loading Flask and assigning the model variable
app=Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')


from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
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
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def main():
    return render_template('index.html')

#receiving the input url from the user and using web scrapping to extract the news content
@app.route('/predict', methods=['GET','POST'])
def predict():
    url=request.get_data(as_text=True)[5:]
    url=urllib.parse.unquote(url)
    article=Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news=article.summary
    
    #passing the news article to the model and returning whether it is fake or real
    pred=model.predict([news])
    return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))

if __name__=='__main__':
    app.run(debug=True)
            


