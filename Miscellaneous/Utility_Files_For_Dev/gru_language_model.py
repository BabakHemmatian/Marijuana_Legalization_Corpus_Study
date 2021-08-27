import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from collections import Counter
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import argparse


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument("--idx",
                    type = int,
                    default = 0)
    
    args = CLI.parse_args()
    print(args)

    machine = args.machine
    idx = args.idx

    #machine = 'alex'
    n_gru_vec = [64, 128, 256, 512]
    n_gru = n_gru_vec[idx]
    epochs = 10
    batch_size = 32
    validation_split = 0.2

    if machine == 'ccv':
        path = '/users/afengler/data/nlp/nietzsche.txt'

    if machine == 'alex' or machine == 'babak':
        path = 'data_files/nietzsche.txt'
    #get_file('nietzsche.txt', origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')

    with open(path, encoding = 'utf-8') as f:
        raw_text = f.read()


    print('corpus length:', len(raw_text))
    print('example text:', raw_text[:150])

    # ideally, we would save the cleaned text, to prevent
    # doing this step every single time
    tokens = raw_text.replace('--', ' ').split()
    cleaned_tokens = []

    for word in tokens:
        word = re.sub('[^A-Za-z0-9]+', '', word)
        if word.isalpha():
            cleaned_tokens.append(word.lower())

    print('sampled original text: ', tokens[:10])
    print('sampled cleaned text: ', cleaned_tokens[:10])

    # build up vocabulary,
    # rare words will also be considered out of vocabulary words,
    # this will be represented by an unknown token
    min_count = 2
    unknown_token = '<unk>'
    word2index = {unknown_token: 0}
    index2word = [unknown_token]

    filtered_words = 0
    counter = Counter(cleaned_tokens)
    # A counter is a container that stores elements as dictionary keys, and their counts are stored as dictionary values.
    for word, count in counter.items():
        if count >= min_count:
            index2word.append(word)
            word2index[word] = len(word2index)
        else:
            filtered_words += 1

    num_classes = len(word2index) # vocabulary size
    print('vocabulary size: ', num_classes)
    print('filtered words: ', filtered_words)

    # create semi-overlapping sequences of words with
    # a fixed length specified by the maxlen parameter
    step = 3 # jumps over this many words to get the next sequence
    maxlen = 40 # won't consider context beyond this
    X = [] # input
    y = [] # output (next word for each sequence)

    for i in range(0, len(cleaned_tokens) - maxlen, step): # stop taking steps once you're "maxlen" words away from the end of the text
        sentence = cleaned_tokens[i:i + maxlen]
        next_word = cleaned_tokens[i + maxlen] # the word after the 40-word span
        X.append([word2index.get(word, 0) for word in sentence]) # get the index from the dictionary for every word
        y.append(word2index.get(next_word, 0)) # same for the word to be predicted

    # keras expects the target to be in one-hot encoded format,
    X = np.array(X) # each row is a 40-word span
    Y = to_categorical(y, num_classes) # creates as many output categories as there are unique "next words" after each span
    print('sequence dimension: ', X.shape)
    print('target dimension: ', Y.shape)
    print('example sequence:\n', X[0])

    embeddings_index = {}
    if machine == 'ccv':
        f = open('/users/afengler/data/nlp/glove.6B.50d.txt')

    if machine == 'alex' or machine == 'babak':
        f = open('data_files/glove.6B.50d.txt')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word2index) + 1, 50))
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word2index) + 1,
                                50,
                                weights = [embedding_matrix],
                                input_length = maxlen,
                                trainable = False)

                                # define the network architecture: embedding, followed by GRU and FF layers


    model = Sequential([
        embedding_layer,
        GRU(n_gru),
        Dense(num_classes, activation = 'softmax'),
    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    print(model.summary())

    def elapsed(sec):
        """
        Converts elapsed time into a more human readable format.

        Examples
        --------
        from time import time

        start = time()
        # do something that's worth timing, like training a model
        elapse = time() - start
        elapsed(elapse)
        """
        if sec < 60:
            return str(sec) + ' seconds'
        elif sec < (60 * 60):
            return str(sec / 60) + ' minutes'
        else:
            return str(sec / (60 * 60)) + ' hours'

    def build_model(model, address = None):
        """
        Fit the model if the model checkpoint does not exist or else
        load it from that address.
        """
        if address is not None or not os.path.isfile(address):
            stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, 
                                patience = 5, verbose = 1, mode = 'auto')
            save = ModelCheckpoint(address, monitor = 'val_loss', 
                                verbose = 0, save_best_only = True)
            callbacks = [stop, save]

            start = time()
            history = model.fit(X, Y, batch_size = batch_size, 
                                epochs = epochs, verbose = 1,
                                validation_split = validation_split,
                                callbacks = callbacks)
            elapse = time() - start
            print('elapsed time: ', elapsed(elapse))
            model_info = {'history': history, 'elapse': elapse, 'model': model}
        else:
            model = load_model(address)
            model_info = {'model': model}

        return model_info

    if machine == 'ccv':
        address1 = '/users/afengler/git_repos/Marijuana_legalization_corpus_study/keras_models/gru_language_pred_seq/gru_weights' + '_' + str(idx) + '.h5'
    if machine  == 'alex':
        address1 = '/users/afengler/OneDrive/git_repos/marijuana_legalization_corpus_study/keras_models/gru_language_pred_seq/gru_weight' + '_' + str(idx) + '.h5'
    if machine == 'babak':
        # PUT YOUR FOLDER STRUCTURE HERE
        pass

    print('model checkpoint address: ', address1)
    model_info1 = build_model(model, address1)

    def check_prediction(model, num_predict):
        true_print_out = 'Actual words: '
        pred_print_out = 'Predicted words: '
        for i in range(num_predict):
            x = X[i]
            prediction = model.predict(x[np.newaxis, :], verbose = 0)
            index = np.argmax(prediction)
            true_print_out += index2word[y[i]] + ' '
            pred_print_out += index2word[index] + ' '

        print(true_print_out)
        print(pred_print_out)


    num_predict = 10
    model = model_info1['model']
    check_prediction(model, num_predict)