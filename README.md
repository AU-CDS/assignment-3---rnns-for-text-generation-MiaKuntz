# Assignment 3 - Language modelling and text generation using RNNs
This assignment focuses on generating text using a trained model and RNNs. The model will be trained on comments from articles published by The New York Times, and the objective is to first create a script where the model is trained, and then create a separate script for generating text based on suggested prompts. 

## Tasks
The tasks for this assignment are to:
-	Train a model for NLP by using TensorFlow.
-	Load the model and generate text.
-	Structure the repository in an appropriate manner.
-	Document my work and workflow for reproducibility.

## Repository content
The GitHub repository contains four folders, namely the ```models``` folder, which contains the trained model and tokenizer, the ```out``` folder, which contains the generated text, the ```src``` folder, which contains the Python scripts for training the model and generating text, and the ```utils``` folder, which contains helper functions for prepping the data, and functions for creating the model and generating text. Additionally, the repository has a ```ReadMe.md``` file, as well ```setup.sh``` and ```requirements.txt``` files.

The models in the ```models``` folder and the generated text in the ```out``` folder was produced by running the script on a small sample set of the data, and are kept in the folder to show a valid example of the output from the “rnn.py” script. 

### Repository structure
| Column | Description|
|--------|:-----------|
| ```models``` | Folder containing the trained model and tokenizer |
| ```out``` | Folder containing the generated text |
| ```src```  | Folder containing Python scripts for training the model and generating text |
| ```utils```  | Folder containing utility functions for prepping the data, and functions for creating the model and generating text |

## Data
The corpus used for this assignment is from the New York Times and consists of articles and comments on articles throughout select months and years. This assignment takes use of the comments files. To download the data, please follow this link:

https://www.kaggle.com/datasets/aashita/nyt-comments

To access and prepare the data for use in the script, please; Create a user on the website, download the data, and move it to the repository. It should appear as a folder called ```archive```, which is what will be used in the scripts. 

## Methods
The following is a description of parts of my code where additional explanation of my decisions on arguments and functions may be needed than what is otherwise provided in the code. 

To generate text, I first need to train a model. This model will be trained with data, which has first been processed by filtering away any parts of the dataset which are not “Comments”. The data is then cleaned and tokenized, where the total number of every appearing word is added to it, and is lastly used to generate input sequences needed to obtain the padding sequences for the model. The model is then created and trained and will be saved, along with the tokenizer, as these are needed to generate text. When saving the model, the max sequence length of the model is added to the name, since this value is needed in the script for generating text, and will run automatically.  

To generate text, I first add argparse arguments to define the different prompts I wish the text to be defined by. I then load the previously saved model and tokenizer, and uses these, along with the maximum sequence length and argparse argument, to generate text. The generated text will be saved in a text file in the ```out``` folder, where it also saves the specific arguments for that generated text. 

## Usage
### Prerequisites and packages
To be able to reproduce and run this code, make sure to have Bash and Python3 installed on whichever device it will be run on. Please be aware that the published scripts were made and run on a MacBook Pro from 2017 with the MacOS Ventura package, and that all Python code was run successfully on version 3.11.1.

The repository will need to be cloned to your device. Before running the code, please make sure that your Bash terminal is running from the repository; Afterwards, please run the following from the command line to install and update the necessary packages:

    bash setup.sh

### Running the scripts
My system requires me to type “python3” at the beginning of my commands, and the following is therefore based on this. To run the scripts from the command line please be aware of your specific system, and whether it is necessary to type “python3”, “python”, or something else in front of the commands. As the text_gen.py script uses argparse to generate text, please be aware to include: the filename of the trained model after --filename, the start word for the generated text after --start_word, and the number of words, as a number, in the generated text after --length when running. Now run:

    python3 src/rnn.py

And:

    python3 src/text_gen.py --filename --start_word --length

This will activate the scripts. When running, it will go through each of the functions in the order written in my main functions. That is:
-	For the ```rnn.py``` script:
o	Processing in the way of reading and cleaning the dataset, initializing and fitting tokenizer, and getting input and padding sequences. 
o	Training the model.
o	Saving the tokenizer and model to the ```models``` folder.
-	And for the ```text_gen.py``` script:
o	Adding argparse arguments.
o	Loading model and tokenizer from ```models``` folder.
o	Saving the generated text to the ```out``` folder.

## Results
The two scripts in this repository work together to generate text based on a model trained on comments on articles by The New York Times. The text is generated based on three arguments: filename, start_word, and length. The chosen arguments when running the command line are saved along with the generated text in a text file saved to the ```out``` folder. It is here possible to see all generated text. I have chosen to keep three examples of generated text as a way of showing a successful output of the scripts. These examples vary in start_words and length to demonstrate what the code can adjust depending on the argument prompt. I once again wish to emphasize that these examples were generated based on a model trained on a very small sample of the dataset.

