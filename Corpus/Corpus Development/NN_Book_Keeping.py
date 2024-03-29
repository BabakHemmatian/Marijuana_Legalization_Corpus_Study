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
# # NOTE: feed in [human_ratings_pattern] as an argument if there are previous
# samples that you would like to be excluded in the next sampling. The fn uses
# glob to match the list of patterns provided to files within the offered path
# and include them in training/testing
# NOTE: Make sure the corresponding "info" files containing the previous samples'
# metadata are stored in the same directory
theparser.Rel_sample(human_ratings_pattern = ["/auto_labels/sample_info-200-False-*"])

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

### call the function for calculating the percentage of relevant comments
# NOTE: May work only for full-year sets of dates
if calculate_perc_rel:
    theparser.Perc_Rel_RC_Comment()
