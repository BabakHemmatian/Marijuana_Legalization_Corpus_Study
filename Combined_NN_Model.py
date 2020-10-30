### import the required modules and functions

import time
import sys
from Utils import *
from config import *
from ModelEstimation import NNModel
from reddit_parser import Parser # Does the parser object need to be adjusted?

# QUESTION: Does the ID need to show up here in the functions too?
theparser=Parser()

# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()

### check key hyperparameters for correct data types
NN_param_typecheck()

### Define the neural network object
nnmodel=NNModel()

### create training, development and test sets

# NOTE: Always index training set first.
# NOTE: For valid analysis results, maximum vocabulary size and frequency filter
# should not be changed between the creation of sets for LDA and NN analyses

## Determine the comments that will comprise various sets
# NOTE: For DOI training, feed in [human_ratings_pattern] as an argument. The fn
# uses glob to match the list of patterns provided to files within the offered
# path and include them in training/testing
# NOTE: Make sure the corresponding "info" files with the ratings' metadata
# are stored in the same directory.
# NOTE: The prefix will be automatically set to [model_path]
nnmodel.Define_Sets(human_ratings_pattern = ["/Ratings/sample_ratings-200-False-*","/Ratings/1-rel_sample_ratings-300-False-*"])

## Read and index the content of comments in each set
# TODO: Set this up to add the RoBERTa activations as flattened column to the database
nnmodel.RoBERTa_Set()

#TODO: Add the getting sentiments and human ratings as part of the training func,
# wuth path arguments if needed

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

# TODO: implement
nnmodel.test()
