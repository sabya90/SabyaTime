import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import os
import keras
import preprocessor as p
from textblob import TextBlob

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

@app.route('/tweet/')
def tweet():
    return render_template('tweet.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    for x in request.form.values():
        print(x)
    line = x
    #print(xx)
    #lines = x.split('\r\n')
    lines = []
    
    lines.extend(re.split('[?.!\r\n]', line.lstrip()))
    lines = [a.strip() for a in lines]
    lines = list(filter(None,lines))
    lines = list(filter(str.strip, lines))
    lines = [item for item in lines if len(item.split())>1]
    #print(lines)
    if len(lines)<3:
        return render_template('index.html', lerror='Please write atleast three proper sentences!', sen=x)
    elif len(lines)>50:
        return render_template('index.html', lerror='Number of Sentences Exceed current Support!')
    #print(y)
    #line = re.sub(' +', ' ', y)
    else:
        #print(lines)
        #int_features = [int(x) for x in request.form.values()]
        #final_features = [np.array(int_features)]
        #prediction = model.predict(final_features)

        #training data processing
        texts = []
        valence = []
    
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
            if TextBlob(text.get_text()).sentiment.polarity > 0.3:
                valence.append('Positive')
            elif TextBlob(text.get_text()).sentiment.polarity < 0.0:
                valence.append('Negative')
            else:
                valence.append('Neutral')
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

        #Valence
        val = max(set(valence), key = valence.count)

        #Calculating Temporal Profile
        total = len(lines)
        pp = 0
        pn = 0
        pnt = 0
        prp = 0
        prn = 0
        prnt = 0
        fp = 0
        fn = 0
        fnt = 0
        if len(valence)!=len(lines):
            print('ERROR!!!')
        for j in range(0, len(valence)):
            if predict_classes[j] == 0 and valence[j]=='Positive':
                pp = pp + 1
            elif predict_classes[j] == 0 and valence[j]=='Negative':
                pn = pn + 1
            elif predict_classes[j] == 0 and valence[j]=='Neutral':
                pnt = pnt + 1
            elif predict_classes[j] == 1 and valence[j]=='Positive':
                prp = prp + 1
            elif predict_classes[j] == 1 and valence[j]=='Negative':
                prn = prn + 1
            elif predict_classes[j] == 1 and valence[j]=='Neutral':
                prnt = prnt + 1
            elif predict_classes[j] == 2 and valence[j]=='Positive':
                fp = fp + 1
            elif predict_classes[j] == 2 and valence[j]=='Negative':
                fn = fn + 1
            elif predict_classes[j] == 2 and valence[j]=='Neutral':
                fnt = fnt + 1
        pp_ratio = "{:.2f}".format((pp/total)*100)
        pn_ratio = "{:.2f}".format((pn/total)*100)
        pnt_ratio = "{:.2f}".format((pnt/total)*100)
        prp_ratio = "{:.2f}".format((prp/total)*100)
        prn_ratio = "{:.2f}".format((prn/total)*100)
        prnt_ratio = "{:.2f}".format((prnt/total)*100)
        fp_ratio = "{:.2f}".format((fp/total)*100)
        fn_ratio = "{:.2f}".format((fn/total)*100)
        fnt_ratio = "{:.2f}".format((fnt/total)*100)

        tp_list = ['Past Positive: '+str(pp_ratio), 'Past Negative: '+str(pn_ratio), 'Past Neutral: '+str(pnt_ratio), 'Present Positive: '+str(prp_ratio), 'Present Negative: '+str(prn_ratio), 'Present Neutral: '+str(prnt_ratio), 'Future Positive: '+str(fp_ratio), 'Future Negative: '+str(fn_ratio), 'Future Neutral: '+str(fnt_ratio)]

        return render_template('index.html', prediction_text=predict_classes, length=int(len(predict_classes)), sen2=lines, name='name', val=val, tp_list=tp_list)

@app.route('/predict_tweet',methods=['POST'])
def predict_tweet():
    '''
    For rendering results on HTML GUI
    '''
    for x in request.form.values():
        print(x)
    
    if os.path.exists('file.txt'):
        os.remove('file.txt')

    os.system('python3 tweep.py -u '+x+' --limit 500 -o file.txt')
    texts = []
    valence = []
    if not os.path.exists('file.txt'):
        #print('wrong username')
        return render_template('tweet.html', error='error', hname=x)
    else:
        y = x
        file = open('file.txt', 'r')
        for line in file:
            text = line.split('<'+y+'>')[1].strip()
            text = p.clean(text)
            text = text.lower().replace('[^\w\s]',' ').replace('\s\s+', ' ')
            if len(text.split())>2:
                #print(text)
                text = clean_str(text)
                texts.append(text)
                if TextBlob(text).sentiment.polarity > 0.3:
                    valence.append('Positive')
                elif TextBlob(text).sentiment.polarity < 0.0:
                    valence.append('Negative')
                else:
                    valence.append('Neutral')

        tokenizer = pickle.load(open('tokenizer-train.pkl','rb'))
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        MAX_SEQUENCE_LENGTH = 31
    
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        indices = np.arange(data.shape[0])
        data = data[indices]
        x_test = data[:]
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
    
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
 
        prediction = loaded_model.predict(x_test)
        predict_classes = prediction.argmax(axis=-1)
    
        #output = round(prediction[0], 2)
        #Valence
        #Valence
        val = max(set(valence), key = valence.count)
        return render_template('tweet.html', prediction_text2=predict_classes, length2=int(len(predict_classes)), user=y, xval=val)

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host = '0.0.0.0', port = 8080)
