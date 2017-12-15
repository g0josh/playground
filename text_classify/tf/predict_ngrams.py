#!/usr/bin/env python2

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import pickle
from model import getModel
import sys
from math import floor
import argparse
import json

DATA_PATH = '/home/cbarobotics/dev/playground/text_classify/tf/'
MODEL_FILE = DATA_PATH + 'model/model_ngrams.tflearn'
WORDS_FILE = DATA_PATH + 'model/words_data_ngrams.pkl'
IGNORE_WORDS = ['?']


def getNGrams(word_list, n):
    return zip(*[word_list[i:] for i in range(n)])

def prepareDataForTf(words, sentence):
    sentence_words = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if stemmer.stem(w.lower()) not in IGNORE_WORDS]
    sentence_ngrams = getNGrams(sentence_words, 2)
    bag = []
    for ngram in words:
        bag = bag+[1] if ngram in sentence_ngrams else bag+[0]

    return(np.array([bag]))

def setUpModel():
    tf.reset_default_graph()
    model = getModel(words_data['input_tensor_length'], words_data['output_tensor_length'])
    model = tflearn.DNN(model, tensorboard_dir='tflearn_logs')
    model.load(MODEL_FILE)

    return model

def predict(sentence, words_data):
    input_tensor = prepareDataForTf(words_data['words'], sentence)
    _result = model.predict(input_tensor)[0]
    classes = words_data['classes']
    result = {}
    for index, _class in enumerate(classes):
        result[_class] = floor(_result[index]*100)
        # print "{}:{:2f}%".format(_class, result[index]*100)
    print "\nSentence:{}\nResult = {}".format(sentence, result)

if __name__ == '__main__':
    # sentence = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', '-s', type=str)
    parser.add_argument('--filename', '-f', type=str)
    args = parser.parse_args()

    with open(WORDS_FILE, 'r') as pkl:
        words_data = pickle.load(pkl)
    sentence = None
    file = None
    if args.sentence:
        sentence=args.sentence
    elif args.filename:
        file = args.filename

    if sentence is None and file is None:
        print "Enter a sentence or a filename"
    else:
        model = setUpModel()
        if sentence is not None:
            predict(sentence, words_data)
        elif file is not None:
            print 'Opening file - {}'.format(file)
            try:
                with open(file, 'r') as f:
                    json_file = json.load(f)
                for sentence in json_file['testing_data']:
                    predict(sentence, words_data)
            except Exception as e:
                print "Exception while opening test file = {}".format(e)

