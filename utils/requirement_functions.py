#!/usr/bin/env python
# data processing tools
import string
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# defining clean text function
def clean_text(txt):
    # making comments lowercase, removing punctuation and creating spaces
    txt = "".join(v for v in txt if v not in string.punctuation).lower() 
    # encoding txt
    txt = txt.encode("utf8").decode("ascii",'ignore') 
    return txt 

# defining sequence of tokens function
def get_sequence_of_tokens(tokenizer, corpus):
    # creating empty list
    input_sequences = []
    # looping over each line in corpus
    for line in corpus:
        # convertimg text into list of integer tokens using tokenizer
        token_list = tokenizer.texts_to_sequences([line])[0]
        # generating loop range from 1 up to length of token list
        for i in range(1, len(token_list)):
            # generating n-gram sequence
            n_gram_sequence = token_list[:i+1]
            # appending n-gram sequence to input_sequences
            input_sequences.append(n_gram_sequence)
    return input_sequences

# defining pad sequences function
def generate_padded_sequences(input_sequences, total_words):
    # getting length of longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # making every sequence length of longest sequence by padding
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))
    # creating predictors and label using input_sequences
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    # converting vector to binary class matrix
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

# defining create model function
def create_model(max_sequence_len, total_words):
    # defining input length
    input_len = max_sequence_len - 1
    # defining model
    model = Sequential()
    # adding input embedding layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    # adding hidden layer 1 - LSTM layer
    model.add(LSTM(100))
    # removing 10% of weight when model is training
    model.add(Dropout(0.1))
    # adding output layer
    model.add(Dense(total_words, 
                    activation='softmax'))
    # adding optimizer
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    return model

# defining generate text function
def generate_text(tokenizer, seed_text, next_words, model, max_sequence_len):
    # creating iterating loop
    for _ in range(next_words):
        # tokenizing text into list of tokens
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # padding list of tokens
        token_list = pad_sequences([token_list], 
                                    maxlen=int(max_sequence_len)-1, 
                                    padding='pre')
        # predicting next word
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)   
        # initializing empty string
        output_word = ""
        # creating iterating loop
        for word,index in tokenizer.word_index.items():
            # checking if current word in index matches predicted
            if index == predicted:
                # assigning word to output_word
                output_word = word
                break
        # appending output_word to seed_text
        seed_text += " "+output_word
    return seed_text.title()