# BUG: Need to deal with the case where after CoreNLP fails, there is no sentence
# left.
# TODO: ignored sentences should also be accounted for. That's different from
# corenlp crashing and seems to happen more often
# TODO: increase memory request: some ignored sentences are rather short
# Delete the last three months on CCV

# QUESTION: How to determine the number of batches?
# NOTE: To simplify the coding, I should just feed in consecutive IDs of each 24 months
# through the sbatch file. In other words:
# The batch IDs should be determined as
# follows: 0 for (2008,1), then +1 for each month after.

import subprocess
import time
import sys
from pycorenlp import StanfordCoreNLP
from reddit_parser import Parser
from defaults import num_process

### call the parsing function
# CoreNLP
# create a connection to the CoreNLP server to retrieve sentiment
# (requires CoreNLP_server.py in the same directory)
subprocess.Popen(
    ['java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -threads ' + str(num_process) + ' --quite'],
    shell=True, cwd="./stanford-corenlp-4.0.0")
time.sleep(5)  # wait for connection to the server to be established

# QUESTION: Does the ID need to show up here in the functions too?
theparser = Parser()

# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()
