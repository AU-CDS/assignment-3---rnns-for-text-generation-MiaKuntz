# importing helper functions
import os 
import sys
sys.path.append(".")
import utils.requirement_functions as rf
# keras module 
import tensorflow as tf
# argparse
import argparse
# joblib
from joblib import load

# defining parsers
def input_parse():
    # initializing parser
    parser = argparse.ArgumentParser()
    # adding arguments
    parser.add_argument("--filename", type=str)
    parser.add_argument("--start_word", type=str, default="danish")
    parser.add_argument ("--length", type=int, default=5)
    # parsing arguments from command line
    args = parser.parse_args()
    return args

# defining models function
def load_models(args):
    # defining filename
    filename = args.filename
    # importing trained model
    loaded_model = tf.keras.models.load_model(f"models/{filename}")
    # getting max sequence length from filename
    max_sequence_len = filename.split("_")[1].split("q")[1].split(".")[0]
    # setting tokenizer path
    tokenizer_path = os.path.join("models/tokenizer.joblib")
    # loading tokenizer
    tokenizer = load(tokenizer_path)
    # return arguments
    return loaded_model, max_sequence_len, tokenizer

# creating the main function
def main():
    # arguments
    args = input_parse()
    # processing file
    loaded_model, max_sequence_len, tokenizer = load_models(args)
    # generating text 
    print(rf.generate_text(tokenizer, args.start_word, args.length, loaded_model, max_sequence_len))

if __name__=="__main__":
    main()