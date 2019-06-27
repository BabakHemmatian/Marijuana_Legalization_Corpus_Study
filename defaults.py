import nltk
import os
import re
import sys

# NOTE: This file only contains the hyperparameters at the most abstract level,
# those that are most likely to be tuned by the user. See relevant functions
# in Utils.py for finer tuning of parameters.
# NOTE: If you wish to keep the following values as defaults, but try out other
# values, you can override the defaults by assinging variables in lda_config.py

### determine hyperparameters ###

### Model choice hyperparameters

NN = False # For development purposes. Should always be set to False for LDA
ENTIRE_CORPUS = True # Are we using a random subset of comments, or the whole
# dataset? The names of output files will include the value of this variable
OVERWRITE = False # Overwrites existing sampled comment indices. Only matters
# if ENTIRE_CORPUS = False
DOWNLOAD_RAW = True # If a raw data file is not available on disk, download it
# NOTE: Be mindful of possible changes to compression algorithm used at
# https://files.pushshift.io/reddit/comments/ beyond 02-2019, as they would
# not be reflected in the parser's code
CLEAN_RAW = True # After parsing, delete the raw data file from disk if it was
# not downloaded during parsing
vote_counting = True # Count number of upvotes when parsing
WRITE_ORIGINAL = True # Write original comments to file when parsing
author = True # Write the username of each post's author to a separate file
# add this function
sentiment = True # Write the average sentiment of a post to file

### Pre-processing hyperparameters
MaxVocab = 2000000 # maximum size of the vocabulary
FrequencyFilter = 1 # tokens with a frequency equal or less than this number
# will be filtered out of the corpus
no_below = 5 # tokens that appear in less than this number of documents in
# corpus will be filtered out
no_above = 0.99 # tokens that appear in more than this fraction of documents in
# corpus will be filtered out
training_fraction = 0.99 # what percentage of data will be used for training.
# The rest of the dataset will be used as an evaluation set for calculating
# perplexity
NN_training_fraction = 0.80 # fraction of the data that is used for training
# the neural network.[1 - training_fraction] fraction of the dataset will be
# divided randomly and equally into evaluation and test sets
calculate_perc_rel = True # whether the percentage of relevant comments from
# each year should be calculated and written to file
# Should be set to False if including 3-2018 or later, as the source does
# not report numbers for those months

### LDA hyperparameters
# TODO: Change number of processes from manually set into a hyperparameter
n_random_comments = 1500 # number of comments to sample from each year for
# training. Only matters if ENTIRE_CORPUS = False.
iterations = 1000 # number of times LDA posterior distributions will be sampled
num_threads = 5 # number of threads used for parallelized processing of comments
# Only matters if using _Threaded functions
num_topics = 50 # number of topics to be generated in each LDA sampling
alpha = 0.1 # determines how many high probability topics will be assigned to a
# document in general (not to be confused with NN l2regularization constant)
minimum_probability = 0.01 # minimum acceptable probability for an output topic
# across corpus
eta = 0.1 # determines how many high probability words will be assigned to a
# topic in general
minimum_phi_value = 0.01 # determines the lower bound on per-term topic
# probability. Only matters if per_word_topics = True.
calculate_perplexity = True # whether perplexity is calculated for the model
calculate_coherence = True # whether umass coherence is calculated for the model

### Neural Network Hyperparameters

## determine kind of network

special_doi = False # If False, the neural network will model sentiment.
# If true, it will perform classification on comments based on a user-defined
# "dimension of interest"
pretrained = False # whether there is sentiment analysis pre-training. Can only be set to True if classifier is also True
# NOTE: For classifier pretraining, the code should first be run with
# special_doi = False & pretrained = False and param_path should be set
# according to the output_path that results from the first run of the code
# NOTE: If pre-training is on, network hyperparameters should not be changed for
# the DOI run from the ones used for pre-training #TODO: Remove this requirement
LDA_topics = True # whether the neural networks take as part of their input
# topic contributions to each post as determined by a previously-run LDA
# NOTE: the path to the LDA output to be used needs to be entered below manually
authorship = True # whether the neural networks take as part of their input
# the username of a post's author.
# NOTE: When the username is missing for posts (e.g. in case the author deleted
# their Reddit account), the model assumes a different author for each anonymous
# posting
# NOTE: The functions assume that "author" files from pre-processing are
# available in the same folder as the one containing this file

## Training hyperparameters
epochs = 3 # number of epochs
learning_rate = 0.003 # learning rate #TODO: write code for automatically testing a set of learning rates
batchSz = 50 # number of parallel batches
embedSz = 128 # embedding size
hiddenSz = 512 # number of units in the recurrent layer
ff1Sz = 1000 # number of units in the first feedforward layer
ff2Sz = 1000 # number of units in the second feedforward layer
keepP = 0.5 # 1 - dropout rate
early_stopping = True # whether to stop training if development set perplexity is going up
l2regularization = False # whether the model will be penalized for longer weight vectors. Helps prevent overfitting
NN_alpha = 0.01 # L2 regularization constant

### Sampling hyperparameters
top_topic_set = None # Choose specific topics to sample comments and top words
# from. set to None to use threshold or fraction instead
sample_topics = None # percentage of topics that will be selected for reporting
# based on average yearly contribution. Set to None if choosing topics based on
# threshold instead
top_topic_thresh = 0.03 # threshold for proportion contribution to the corpus
# determining topics to report
topn = 80 # the number of high-probability words for each topic to be exported
# NOTE: Many of the words will inevitably be high probability general
# non-content and non-framing words. So topn should be set to significantly
# higher than the number of relevant words you wish to see
sample_comments = 25 # number of comments that will be sampled from top topics
min_comm_length = 20 # the minimum acceptable number of words in a sampled
# comment. Set to None for no length filtering
# Determines how topic contributions are calculated. When set to True, the
# topic of each word is set to be simply the most probable topic. When False,
# the topic of each word is set to the entire probability distribution over
# num_topics topics.
one_hot_topic_contributions=True
# BUG: non-one-hot topic contribution phi-value calculation gives an assertion
# error
topic_cont_freq="monthly" # The frequency of topic contribution calculation
num_pop = 2000 # number of the most up- or down-voted comments sampled for model
# comparison. Set to None for no sampling. Needs prior parsing with
# write_original = True

### Paths

## where the data is
# NOTE: if not fully available on file, set Download for Parser function to
# True (source: http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable
# accordingly
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)

## Year/month combinations to get Reddit data for
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=[2008] # year range
for year in years:
    for month in months:
        if year==2019: # until Jan 2019
            break
        dates.append((year,month))

## where the output will be stored
# NOTE: To avoid confusion between different kinds of models, record the
# variables most important to your iteration in the folder name

## where the output will be stored

# NOTE: To avoid confusion between different kinds of models, always include doi\
# and pre in the output directory's name. After those, record the variables most
# important to your iteration

if NN: # If running a neural network analysis
    output_path = path+"/"+"doi_"+str(classifier)+"_pre_"+str(pretrained)+"_e_"+str(epochs)+"_"+"hd"+"_"+str(hiddenSz)
    if not os.path.exists(output_path):
        print("Creating directory to store the output")
        os.makedirs(output_path)

    ## where the saved parameters are

    # NOTE: Enter manually. Only matters if special_doi = True and pretrained = True

    param_path = path+"/doi_False_pre_False_e_3_hd_512/"
    if pretrained == True:
        if not os.path.exists(param_path):
            raise Exception("Could not find saved pre-trained parameter values.")

else: # if doing topic modeling

    # Force this import so output_path is correctly set
    from lda_config import ENTIRE_CORPUS
    output_path = path + "/LDA_"+str(ENTIRE_CORPUS)+"_"+str(num_topics)

### Preprocessing ###

### determine the set of stopwords used in preprocessing

keepers = ["how","should","should've","could","can","need","needn","why","few",
"more","most","all","any","against","because","ought","must","mustn","mustn't",
"shouldn","shouldn't","couldn't","couldn","shan't", "needn't"]
stop = []
for word in set(nltk.corpus.stopwords.words('english')):
    if word not in keepers:
        stop.append(str(word))

### Define the regex filter used for finding relevant comments

# get the list of words relevant to legality from disk
# (requires legality.txt to be located in the same directory)
legality = []
with open("legality.txt",'r') as f:
    for line in f:
        legality.append(re.compile(line.lower().strip()))
# get the list of words relevant to marijuana from disk
# (requires marijuana.txtto be located in the same directory)
marijuana = []
with open("marijuana.txt",'r') as f:
    for line in f:
        marijuana.append(re.compile(line.lower().strip()))
