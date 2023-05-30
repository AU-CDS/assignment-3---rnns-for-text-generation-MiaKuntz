# Author: Mia Kuntz
# Date hand-in: 31/5 - 2023

# Description: This script is used to train a model for text generation.
# The script saves the trained model and tokenizer to the models folder.

# importing operating system 
import os
# importing pandas
import pandas as pd
# importing numpy
import numpy as np
# setting seed
np.random.seed(42)
# importing tensorflow
import tensorflow as tf
# setting seed
tf.random.set_seed(42)
# importing keras
from tensorflow.keras.preprocessing.text import Tokenizer
# importing helper functions
import sys
sys.path.append(".")
# importing utils functions
import utils.requirement_functions as rf
# importing warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# importing joblib
from joblib import dump

def process_data(): 
    # setting data directory
    data_dir = os.path.join("archive")
    # creating empty list for comments
    all_comments = []
    # looping through files in data directory
    for filename in os.listdir(data_dir):
        # checking if file is a csv file
        if 'Comments' in filename:
            # reading csv file
            comments_df = pd.read_csv(data_dir + "/" + filename)
            # appending comments to list
            all_comments.extend(list(comments_df["commentBody"].values))
    # removing unknown comments
    all_comments = [c for c in all_comments if c != "Unknown"]
    # cleaning text
    corpus = [rf.clean_text(x) for x in all_comments]
    # initializing tokenizer
    tokenizer = Tokenizer()
    # fitting tokenizer
    tokenizer.fit_on_texts(corpus)
    # getting total words 
    total_words = len(tokenizer.word_index) + 1 
    # getting input sequences 
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus)
    # padding sequences 
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences, total_words)
    return tokenizer, total_words, predictors, label, max_sequence_len

# defining function to train model
def model_training(sequence_length, all_words, predict, y):
    # creating model
    model = rf.create_model(sequence_length, all_words)
    # compiling model
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
    # training model
    model = model_training(max_sequence_len, total_words, predictors, label)
    # saving tokenizer
    dump(tokenizer, "models/tokenizer.joblib")
    # creating output path
    outpath = os.path.join(f"models/rnn-model_seq{max_sequence_len}.keras")
    # saving model
    model.save(outpath, overwrite=True, save_format=None)

if __name__=="__main__":
    main()

# Command line argument:
# python3 rnn.py 