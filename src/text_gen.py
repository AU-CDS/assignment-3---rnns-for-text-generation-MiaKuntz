# importing helper functions
import sys
sys.path.append(".")
import utils.requirement_functions as rf
# keras module 
import tensorflow as tf

# importing trained model
loaded_model = tf.keras.saving.load_model("model/rnn_model.keras")

# generating text ????
print(rf.generate_text("danish", 5, loaded_model, max_sequence_len))