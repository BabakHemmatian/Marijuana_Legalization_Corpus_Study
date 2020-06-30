# NOTE: To simplify the coding, just feed in consecutive batch IDs with
# num_process in mind (e.g. if num_process=3, array=0 would process the first
# 3 months in the dates array)

# BUG: Because of a hacky solution within lang_filtering(), the language filtering
# would only work properly for fully consecutive set of months within self.dates
# TODO: make it more general

import subprocess
import time
import sys
from pycorenlp import StanfordCoreNLP
from reddit_parser import Parser
import argparse
import numpy
from defaults import *

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--array', type = int)
    argparser.add_argument('--machine', type = str)
    args = argparser.parse_args()

### call the parsing function

# CoreNLP
# create a connection to the CoreNLP server to retrieve sentiment
# (requires CoreNLP_server.py in the same directory)
subprocess.Popen(
    ['java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -threads ' + str(num_process) + ' --quite'],
    shell=True, cwd="./stanford-corenlp-4.0.0")
time.sleep(5)  # wait for connection to the server to be established

theparser=Parser(array=args.array,machine=args.machine)

# Create relevant folders
theparser.safe_dir_create()

# parse the documents
theparser.Parse_Rel_RC_Comments()
