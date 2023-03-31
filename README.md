# Assignment 3 - Language modelling and text generation using RNNs
The assignment focuses on generating text using a trained model and RNN’s. The model will be trained on comments from articles published by The New York Times, and the objective is to first create a script where I train the model, and then create a separate script for generating text based on suggested prompts. 

## Tasks
The tasks for this assignment are to:

-	Train a model for NLP by using TensorFlow,
-	Load the model and generate text,
-	Structure my repository in an appropriate manner,
-	And document my work and workflow for reproducibility

## Repository content
The GitHub repository consists of three folders; The ```src``` folder, which contains the Python script for the RNN model and training, and the script for loading the model and generating text, the ```models``` folder, which contains the trained model and tokenizer, and the ```in``` folder, where the data should be stored after download. Furthermore, the repository contains a ```ReadMe.md``` file, as well as files for ```setup.sh``` and ```requirements.txt```. 

## Data
The data used for this assignment is several files based on articles and comments on them from The New York Times. To use the data in the “rnn.py” script, please do the following:

-	Download the data via following link:

https://www.kaggle.com/datasets/aashita/nyt-comments

-	Import the data to the ```in``` folder in the repository
-	Optionally, change the name to “news_data” for easier reproducibility, as that is what the folder name will be in the scripts. 

## Technicalities and how to run the script
The script and its code is run in VS Code, and uses Python 3.11.1. 

To run the script please notice, that you will first need to run:

    bash setup.sh

And then call the scripts independently in the command line to run them. 

Please note that the ```rnn.py``` script was run with a sample of the corpus, and it is recommended to do the same when running, as the corpus as a whole is quite extensive when included as a whole. This will of course affect how the model is training, but will still be able to show how the code is running. When saving the file, the max sequence length of the model is added to the name, since this value is needed in the script for generating text, and will run automatically.  

Another note to be aware of is that the ```text_gen.py``` script takes several arguments, e.g. filename to be loaded, or in the ```generate_text``` function. The prompt when generating text  can be modified depending on the prompt wished to be run. The example in the published script have “Danish” and “5” as its default and can, as mentioned, be altered to anything else. 
