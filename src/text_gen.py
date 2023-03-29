# importing helper functions
import os 
import sys
sys.path.append(".")
import utils.requirement_functions as rf
# keras module 
import tensorflow as tf
# argparse
import argparse
# loading tokenizer
from joblib import load

# defining filepath function
def file_path():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--filename")
    # parse the argument from command line
    args = parser.parse_args()
    # defining filename
    filename = args.filename
    # importing trained model
    loaded_model = tf.keras.models.load_model(filename)
    # defining max sequence length
    max_sequence_len = filename.split("_")[1].split("q")[1].split(".")[0]
    # return arguments
    return loaded_model, max_sequence_len

# defining function for generating text
def text_generating(ld_model, max_seq_len):
    tokenizer_path = os.path.join("model/tokenizer.joblib")
    tokenizer = load(tokenizer_path)
    text_gen = rf.generate_text(tokenizer, "danish", 5, ld_model, max_seq_len)
    return text_gen

# creating the main function
def main():
    # processing file
    loaded_model, max_sequence_len = file_path()
    # generating text
    text_gen = text_generating(loaded_model, max_sequence_len)
    print(text_gen)

if __name__=="__main__":
    main()