# loading
from joblib import load
# importing helper functions
import sys
sys.path.append(".")
import utils.requirement_functions as rf

# importing trained model
trained_model = load("filepath here")
# or ???
new_model = tf.keras.models.load_model('saved_model/my_model')

# generating text ????
print(rf.generate_text("danish", 5, trained_model))