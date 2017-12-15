#!/usr/bin/env python2

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from model import getModel

DATA_PATH = '/home/cbarobotics/dev/playground/text_classify/tf/'
MODEL_FILE = DATA_PATH + 'model/model_ngrams.tflearn'
WORDS_FILE = DATA_PATH + 'model/words_data_ngrams.pkl'
TRG_FILE = DATA_PATH + 'training_data.json'
IGNORE_WORDS = ['?']

def getNGrams(word_list, n):
    return zip(*[word_list[i:] for i in range(n)])

def parseData():
    global IGNORE_WORDS, TRG_FILE
    try:
        with open(TRG_FILE, 'r') as df:
            json_data = json.load(df)
    except Exception as e:
        print ('Exception while reading data - ', e)
        return False

    intents = []
    bow = []
    docs = []
    train_data = json_data['training_data']
    for intent in train_data:
        # add intent to the list
        intents = intents+[intent['intent']] if intent['intent'] not in intents else intents

        for sentence in intent['sentences']:
            # Get words
            words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if stemmer.stem(w.lower()) not in IGNORE_WORDS]
            ngrams = getNGrams(words, 2)
            bow += [w for w in ngrams if w not in bow]
            # make a doc
            docs.extend( [(ngrams, intent['intent'])] )

    bow = sorted(list(set(bow)))
    # print 'Intents = {}\nbow = {}\ndocs={}'.format(intents, bow, docs)
    return intents, bow, docs

def prepareDataForTf(classes, bow, docs):
    result_data = []
    for doc in docs:
        bag = []
        class_row = [0] * len(classes)
        for ngram in bow:
            bag = bag+[1] if ngram in doc[0] else bag+[0]
        class_row[classes.index(doc[1])] = 1
        result_data.append([bag, class_row])
    # print 'tf data = {}'.format(result_data)
    return result_data

if __name__ == '__main__':
    classes, all_words, docs = parseData()
    training_data = prepareDataForTf(classes, all_words, docs)
    random.shuffle(training_data)
    training = np.array(training_data)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    # print ("shape = {}\nx = {}\ny = {}".format(training.shape, train_x, train_y))

    tf.reset_default_graph()
    net = getModel(len(train_x[0]), len(train_y[0]))
    # Define model and setup tensorboard
    model = tflearn.DNN(net, tensorboard_dir=DATA_PATH+'tflearn_logs')
    # Start training (apply gradient descent algorithm)
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=False)
    model.save(MODEL_FILE)

    with open(WORDS_FILE,'w') as f:
        pickle.dump({'words':all_words, 'classes':classes, 'input_tensor_length':len(train_x[0]), 'output_tensor_length':len(train_y[0])}, f)





