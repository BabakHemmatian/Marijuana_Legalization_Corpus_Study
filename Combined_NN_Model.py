### import the required modules and functions
#TODO: Is this the correct Write_Performance()? --> NO, it's not. We should
# update it with the parameters important for the NN in Utils.py
# Same for NN_param_typecheck in NN_Utils.py
import subprocess
import time
import sys

from pycorenlp import StanfordCoreNLP

from Utils import Write_Performance
from config import *
from reddit_parser import Parser
from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *

# NOTE: Don't forget to set NN=True in defaults.py before running this file

### Define the neural network object

nnmodel=NNModel()


### check key hyperparameters for correct data types

nnmodel.NN_param_typecheck()

### Write hyperparameters to file. Performance measures will be written to the
# same file after analyses are performed

Write_Performance()

### call the parsing function

# CoreNLP
# create a connection to the CoreNLP server to retrieve sentiment
# (requires CoreNLP_server.py in the same directory)
subprocess.Popen(['java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --quiet'],
               shell=True, cwd="./stanford-corenlp-4.0.0")
time.sleep(5)  # wait for connection to the server to be established

theparser=Parser()
# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()

if Neural_Relevance_Filtering:
    # Use a transformer-based neural network trained on human ratings to prune the
    # dataset from irrelevant posts. Path will default to the Human_Ratings folder
    theparser.Neural_Relevance_Screen()

    # Needs results from Neural_Relevance_Screen
    theparser.Neural_Relevance_Clean()

theparser.lang_filtering() # filter non-English posts

# NOTE: Requires a Neural_Relevance_Screen random sample hand-annotated for accuracy
if eval_relevance:
    theparser.eval_relevance()

### call the function for calculating the percentage of relevant comments
# NOTE: May work only for full-year sets of dates
if calculate_perc_rel:
    theparser.Perc_Rel_RC_Comment()

### create training, development and test sets

# NOTE: Always index training set first.
# NOTE: For valid analysis results, maximum vocabulary size and frequency filter
# should not be changed between the creation of sets for LDA and NN analyses

## Determine the comments that will comprise various sets

nnmodel.Define_Sets()

## Read and index the content of comments in each set

for set_key in nnmodel.set_key_list:
    nnmodel.Index_Set(set_key)

## if performing sentiment pre-training, load comment sentiments from file
if special_doi == False and pretrained == False:
    nnmodel.Get_Sentiment(path)
elif special_doi == True:
    nnmodel.Get_Human_Ratings(path)
    #TODO: Define this function. It should already somehow incorporated into
    # the ModelEstimator

### Sentiment Modeling/DOI Classification Neural Networks

## create the computation graph
# Note: Use only CPU if expecting overly large matrices compared to your
# computer's GPU capacity

nnmodel.Setup_Comp_Graph(device_count={'GPU': 0})

### Train and Test the Neural Network ###

## create a list of comment lengths for each set

nnmodel.Get_Set_Lengths()

### Train and test for the determined number of epochs
# NOTE: The results of the test will be written to the Performance file defined
# in defaults.py

nnmodel.train_and_evaluate()
