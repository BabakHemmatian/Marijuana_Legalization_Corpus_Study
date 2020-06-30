# TODO: 144 / 6 = 24. We should probably have 6 runs of the filtering.
# To simplify the coding, I should just feed in consecutive IDs of each 24 months
# through the sbatch file. In other words:
# The batch IDs should be determined as
# follows: 0 for (2008,1), then +1 for each month after.

# BUG: Because of a hacky solution within Neural_Relevance_Clean(), the function
# would only work properly for fully consecutive set of months within self.dates
# TODO: make it more general

### import the required modules and functions

import subprocess
import time
import sys
from Utils import Write_Performance
from config import *
#from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *
from reddit_parser import Parser # Does the parser object need to be adjusted?

# CoreNLP
# create a connection to the CoreNLP server to retrieve sentiment
# (requires CoreNLP_server.py in the same directory)
subprocess.Popen(
    ['java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -threads ' + str(num_process) + ' --quite'],
    shell=True, cwd="./stanford-corenlp-4.0.0")
time.sleep(5)  # wait for connection to the server to be established

# QUESTION: Does the ID need to show up here in the functions too?
theparser=Parser()

# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()

# Use a transformer-based neural network trained on human ratings to prune the
# dataset from irrelevant posts. Path will default to the Human_Ratings folder
# TODO: should depend on the monthly files now instead of the general one
theparser.Neural_Relevance_Screen()

# TODO: Define sets should also be local and come before batched runs of the NN
# SO: at the end of cleaning
