import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import os
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

#from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.models import model_from_json

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

MAX_SEQUENCE_LENGTH = 0 # maximum sentence length
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    for x in request.form.values():
        print(x)
    #line = x
    #print(line)
    lines = x.split('\r\n')
    #print(y)
    #line = re.sub(' +', ' ', y)
    print(lines)
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #training data processing
    texts = []
    texts1 = []
    labels = []
    
    #Process the user test data
    #data_user = pd.read_csv('data/temporal-test.tsv', sep='\t')
    #print (data_user.shape)
    #print (data_user.shape[0]) # number of rows in traing set
    #total_usertesta = data_user.shape[0]

    #text1 = BeautifulSoup(x, "lxml")
    #print (text1)
    #texts1.append(clean_str(text1.get_text()))
    #print (texts1)
    for idx in range(0, len(lines)):
        text = BeautifulSoup(lines[idx], "lxml")
        #print (text)
        texts.append(clean_str(text.get_text()))
        #labels.append(data_user.temporality[idx])
    
    # Tokenize the data    
    
    #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer = pickle.load(open('tokenizer-train.pkl','rb'))
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    #print(len(sequences))
    
    
    MAX_SEQUENCE_LENGTH = 31
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    #labels = to_categorical(np.asarray(labels))
    #print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    
    #np.random.shuffle(indices)
    data = data[indices]
    #labels = labels[indices]
    
    #user test
    x_test = data[:]
    #y_test = labels[:]
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
     
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(x_test, y_test, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    #test_score, test_acc = loaded_model.evaluate(x_test, y_test, batch_size=100)
    #print('Test Score: %1.4f' % test_score)
    #print('Test Accuracy: %1.4f' % test_acc)
 
    prediction = loaded_model.predict(x_test)
    predict_classes = prediction.argmax(axis=-1)

    sen = []
    for i in range(len(predict_classes)):
        if predict_classes[i]==0:
            sen.append('Sentence-'+str(i+1)+' is Past')
        elif predict_classes[i]==1:
            sen.append('Sentence-'+str(i+1)+' is Present')
        else:
            sen.append('Sentence-'+str(i+1)+' is Future')

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=predict_classes, len=len(predict_classes), sen=lines)


if __name__ == "__main__":
    app.run(debug=True)
