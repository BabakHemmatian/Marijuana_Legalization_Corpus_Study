# To simplify the coding, I should just feed in consecutive IDs of each 24 months
# through the sbatch file. In other words:
# The batch IDs should be determined as follows: 0 for (2008,1), then +1 for
# each month after.

# BUG: Because of a hacky solution within Neural_Relevance_Clean(), the function
# would only work properly for fully consecutive set of months within self.dates
# TODO: make it more general

### import the required modules and functions

import time
import sys
from Utils import Write_Performance
from config import *
#from ModelEstimation import NNModel
from transformers import BertTokenizer
from NN_Utils import *
from reddit_parser import Parser # Does the parser object need to be adjusted?

# NOTE: Feed machine="local" as an argument if not running through the cluster
theparser=Parser()

# Use a transformer-based neural network trained on human ratings to prune the
# dataset from irrelevant posts. Path will default to the Human_Ratings folder
theparser.Neural_Relevance_Screen(batch_size=1200)
