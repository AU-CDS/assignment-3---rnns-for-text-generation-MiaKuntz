# loading
from joblib import load
# importing helper functions
import sys
sys.path.append(".")
import utils.requirement_functions as rf
# keras module 
import tensorflow as tf

# importing trained model
new_model = tf.keras.models.load_model("./out/rnn_model.joblib")

# generating text ????
print(rf.generate_text("danish", 5, new_model, max_sequence_len))