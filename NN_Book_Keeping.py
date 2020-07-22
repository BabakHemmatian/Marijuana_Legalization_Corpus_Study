# TODO: Is this the correct Write_Performance()? --> NO, it's not. We should
# update it with the parameters important for the NN in Utils.py
# Same for NN_param_typecheck in NN_Utils.py

# NOTE: This file should be run only once, before Combined_NN_Model.py and after
# Parse.py and Neural_Filtering.py are run for the entire dataset, repeatedly if
# necessary

import subprocess
import time
import sys
from Utils import Write_Performance
from config import *
from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *
from reddit_parser import Parser # Does the parser object need to be adjusted?

theparser=Parser(machine="local")

# parse the documents
theparser.Parse_Rel_RC_Comments()

# NOTE: Requires a Neural_Relevance_Screen random sample hand-annotated for accuracy
if eval_relevance:
    theparser.eval_relevance()

if Neural_Relevance_Filtering:
    # Needs results from Neural_Relevance_Screen via Neural_Filtering.py
    theparser.Neural_Relevance_Clean()
    # TODO: Clean and eval_relevance should be just run locally. Just do those
    # if the batch is len(inputs)

# NOTE: Don't forget to set NN=True in defaults.py before running this file

### Write hyperparameters to file. Performance measures will be written to the
# same file after analyses are performed

Write_Performance()

if add_sentiment:

    # CoreNLP
    # create a connection to the CoreNLP server to retrieve sentiment
    # (requires CoreNLP_server.py in the same directory)
    subprocess.Popen(
        ['java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer'],
        shell=True, cwd="./stanford-corenlp-4.0.0")
    time.sleep(5)  # wait for connection to the server to be established

    theparser.add_c_sentiment()

### call the function for calculating the percentage of relevant comments
# NOTE: May work only for full-year sets of dates
if calculate_perc_rel:
    theparser.Perc_Rel_RC_Comment()
