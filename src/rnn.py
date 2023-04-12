# data processing tools
import os
import pandas as pd
import numpy as np
np.random.seed(42)
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer
# importing helper functions
import sys
sys.path.append(".")
import utils.requirement_functions as rf
# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# importing random
import random
# joblib
from joblib import dump

# defining function for loading data, preprocessing, tokenizing, and padding function
def process_data(): 
    # creating directory path
    data_dir = os.path.join("in/news_data")
    # creating empty list
    all_comments = []
    # creating for loop for loading data
    for filename in os.listdir(data_dir):
        # choosing only data files with "Comments" in title
        if 'Comments' in filename:
            # read in the chosen files
            comments_df = pd.read_csv(data_dir + "/" + filename)
            # creating list of chosen "commentBody" column
            all_comments.extend(list(comments_df["commentBody"].values))
    # sorting out missing data
    all_comments = [c for c in all_comments if c != "Unknown"]
    # cleaning out data set and creating corpus
    corpus = [rf.clean_text(x) for x in all_comments]
    # tokenizing
    tokenizer = Tokenizer()
    # creating tokens based on corpus
    tokenizer.fit_on_texts(corpus)
    # creating sequence of tokens
    total_words = len(tokenizer.word_index) + 1 
    # transforming tokens into a numerical output
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    # padding sequences
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    return tokenizer, total_words, predictors, label, max_sequence_len

# defining model and training function
def model_training(sequence_length, all_words, predict, y):
    # initializing model
    model = rf.create_model(sequence_length, all_words)
    # fitting model
    model.fit(predict, 
              y, 
              epochs=100,
              batch_size=128, 
              verbose=1) 
    return model

# creating main function
def main():
    # processing data
    tokenizer, total_words, predictors, label, max_sequence_len = process_data()
    # creating and training model
    model = model_training(max_sequence_len, total_words, predictors, label)
    # saving tokenizer
    dump(tokenizer, "models/tokenizer.joblib")
    # saving trained model
    outpath = os.path.join(f"models/rnn-model_seq{max_sequence_len}.keras")
    model.save(outpath, overwrite=True, save_format=None)

if __name__=="__main__":
    main()
