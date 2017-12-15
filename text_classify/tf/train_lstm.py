
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import json
import pickle
from model_lstm import getModel

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

DATA_PATH = '/home/cbarobotics/dev/playground/text_classify/tf/'
MODEL_FILE = DATA_PATH + 'model/model_lstm.tflearn'
WEIGHTS_FILE = DATA_PATH + 'model/weights_lstm.h5'
WORDS_FILE = DATA_PATH + 'model/vocabulary.pkl'
TRG_FILE = DATA_PATH + 'training_data.json'
IGNORE_WORDS = ['?']

def prepareData():
    global IGNORE_WORDS, TRG_FILE
    try:
        with open(TRG_FILE, 'r') as df:
            json_data = json.load(df)
    except Exception as e:
        print ('Exception while reading data - ', e)
        return False

    intents = []
    vocabulary = {}
    y_train = []
    x_train = []
    train_data = json_data['training_data']
    n = 0
    for intent in train_data:
        # add intent to the list
        intents = intents+[intent['intent']] if intent['intent'] not in intents else intents

        for sentence in intent['sentences']:
            # Get words
            words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if stemmer.stem(w.lower()) not in IGNORE_WORDS]
            # Update vocabulary
            for word in words:
                if word not in vocabulary.values():
                    n+=1
                    vocabulary.update({n:word})
            x_train.append([i for i, j in vocabulary.iteritems() for w in words if j==w])
            y_train.append([1 if i==intent['intent'] else 0 for i in intents])

    return intents, vocabulary, x_train, y_train

if __name__ =='__main__':
    classes, vocabulary, x_train, y_train = prepareData()
    # print '\nClasses = {}\nvoca = {}\nLength = {}\nx={}\ny={}'.format(classes, vocabulary,len(vocabulary), x_train, y_train)
    max_x_length = 50
    max_y_length = len(classes)
    x_train = sequence.pad_sequences(x_train, padding='post', truncating='post', maxlen=max_x_length)
    y_train = sequence.pad_sequences(y_train, padding='post', truncating='post', maxlen=max_y_length)
    # print "\nx_train = {}\n\ny_train={}".format(x_train, y_train)
    emb_vec_size = 16
    model=getModel(len(vocabulary)+1, emb_vec_size, max_x_length, max_y_length)
    model.fit(x_train, y_train, epochs=10, batch_size=8)

    model_json = model.to_json()
    with open(MODEL_FILE , 'w') as json_file:
        json_file.write(model_json)

    model_weights = model.save_weights(WEIGHTS_FILE)
