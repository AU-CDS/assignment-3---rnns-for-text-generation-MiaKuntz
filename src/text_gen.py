# Author: Mia Kuntz
# Date hand-in: 31/5 - 2023

# Description: This script is used to generate text from a trained model.
# The script takes three arguments: filename, start_word, and length.
# filename: The filename of the trained model.
# start_word: The word the generated text should start with.
# length: The length of the generated text.
# The script then prints the generated text to the terminal.

# importing operating system 
import os 
# importing helper functions
import sys
sys.path.append(".")
# importing utils functions
import utils.requirement_functions as rf
# importing tensorflow
import tensorflow as tf
# importing argparse
import argparse
# importing joblib
from joblib import load

# defining function to parse input
def input_parse():
    # initializing parser
    parser = argparse.ArgumentParser()
    # adding arguments
    parser.add_argument("--filename", type=str)
    parser.add_argument("--start_word", type=str, default="danish")
    parser.add_argument ("--length", type=int, default=5)
    # parsing arguments
    args = parser.parse_args()
    return args

# defining function to load models
def load_models(args):
    # loading model
    filename = args.filename
    loaded_model = tf.keras.models.load_model(f"models/{filename}")
    # getting max sequence length from filename 
    max_sequence_len = filename.split("_")[1].split("q")[1].split(".")[0]
    # loading tokenizer from models folder
    tokenizer_path = os.path.join("models/tokenizer.joblib")
    tokenizer = load(tokenizer_path)
    return loaded_model, max_sequence_len, tokenizer

# defining function to save generated text to file
def save_text_to_file(text, args):
    # creating output file path
    output_file = os.path.join("out", "generated_text.txt")
    # writing to file
    with open(output_file, "a") as file:
        file.write(f"Arguments: --filename {args.filename} --start_word {args.start_word} --length {args.length}\n")
        file.write(f"Generated Text: {text}\n\n")

# defining main function
def main():
    # parsing input
    args = input_parse()
    # loading models
    loaded_model, max_sequence_len, tokenizer = load_models(args)
    # generating text
    generated_text = rf.generate_text(tokenizer, args.start_word, args.length, loaded_model, max_sequence_len)
    # saving text to file
    save_text_to_file(generated_text, args)

if __name__=="__main__":
    main()

# Command line arguments example:
# python3 text_gen.py --filename "rnn-model_seq198.keras" --start_word "danish" --length 5