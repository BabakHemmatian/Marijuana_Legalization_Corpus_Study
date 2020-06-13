### import the required modules and functions

import subprocess
import time
import sys
from Utils import Write_Performance
from config import *
from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *
from reddit_parser import Parser # Does the parser object need to be adjusted?

# QUESTION: Does the ID need to show up here in the functions too?
theparser=Parser()

# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()


### Define the neural network object

nnmodel=NNModel()

### check key hyperparameters for correct data types

nnmodel.NN_param_typecheck()

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
