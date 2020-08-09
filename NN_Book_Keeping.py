import subprocess
import time
import sys
from Utils import Write_Performance
from config import *
from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *
from reddit_parser import Parser

# define the parser object with variables from defaults.py, imported via config.py
# NOTE: Feed machine="local" as an argument if not running through the cluster
theparser=Parser(machine="local")

# Extract a random sample of rel_sample_num documents along with their labels
# according to the pretrained neural relevance classifier, to evaluate classifier
# performance.
# NOTE: Will skip if a previous sample is found on file to prevent potentially
# overwriting human judgments
theparser.Rel_sample()

# NOTE: Requires rel_sample results hand-annotated for accuracy
# Reports per-class precision, recall, f1 and accuracy
# If there are different evaluation trials with the same rel_sample_num and
# balanced_rel_sample parameters, feed in trial=[trial number] as an argument.
# By default, the naming convention for the rated files are:
# rel_sample_ratings-[rel_sample_num/ num_annot]-[balanced_rel_sample]-
# [an annotator's index]-[trial number], with the last element missing if there
# is only one trial for that combination of parameters
if eval_relevance:
    theparser.eval_relevance(trial=2)

# NOTE: If you haven't ensure classifier accuracy using a random sample, comment
# out the following
if Neural_Relevance_Filtering:
    # Needs results from Neural_Relevance_Screen via Neural_Filtering.py
    theparser.Neural_Relevance_Clean()


### Write hyperparameters to file. Performance measures will be written to the
# same file after analyses are performed

# Write_Performance()
# #
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
