import numpy as np
import json
from model_lstm import getModel

from keras.preprocessing import text, sequence

DATA_PATH = '/Users/josh/Work/playground/text_classify/tf/'
MODEL_FILE = DATA_PATH + 'model/model_lstm.tflearn'
WEIGHTS_FILE = DATA_PATH + 'model/weights_lstm.h5'
WORDS_FILE = DATA_PATH + 'model/vocabulary.pkl'
TRG_FILE = DATA_PATH + 'training_data_sample.json'

def prepareData(vocabulary_size, list_of_intent_dicts):
    x_train, y_train = [], []
    intents = []
    for intent_dict in list_of_intent_dicts:
        intents = intents+[intent_dict['intent']] if intent_dict['intent'] not in intents else intents
        for sentence in intent_dict['sentences']:
            sentence_vec = text.hashing_trick(sentence, 
                                       vocabulary_size,
                                       hash_function='md5',
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
            x_train.append(sentence_vec)
            y_train.append([1 if i==intent_dict['intent'] else 0 for i in intents])

    return intents, x_train, y_train


if __name__ =='__main__':
    # parse json trg file
    try:
        with open(TRG_FILE, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print ("Exception while reading json trg file - {}".format(e))
        exit()

    vocabulary_size = 5000
    classes, x_train, y_train = prepareData(vocabulary_size, json_data['training_data'])
    print ('x_train ={}\n\ny_train={}'.format(x_train, y_train))
    max_x_length = 50
    max_y_length = len(classes)
    x_train = sequence.pad_sequences(x_train, padding='post', truncating='post', maxlen=max_x_length)
    y_train = sequence.pad_sequences(y_train, padding='post', truncating='post', maxlen=max_y_length)
    print ("\nx_train = {}\n\ny_train={}".format(x_train, y_train))

    emb_vec_size = 32
    model=getModel(vocabulary_size, emb_vec_size, max_x_length, max_y_length)
    model.fit(x_train, y_train, epochs=100, batch_size=8)

    model_json = model.to_json()
    with open(MODEL_FILE , 'w') as json_file:
        json_file.write(model_json)

    model_weights = model.save_weights(WEIGHTS_FILE)
