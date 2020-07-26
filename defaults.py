import nltk
import os
import re
import sys
import numpy as np

# NOTE: This file only contains the hyperparameters at the most abstract level,
# those that are most likely to be tuned by the user. See relevant functions
# in Utils.py for finer tuning of parameters.
# NOTE: If you wish to keep the following values as defaults, but try out other
# values, you can override the defaults by assinging variables in lda_config.py

### determine hyperparameters ###

### Data management hyperparameters

NN = True # Set to False for LDA, to true for neural network classification
ENTIRE_CORPUS = True # Are we using a random subset of comments, or the whole
# dataset? The names of model files and output directories will include the
# value of this variable (e.g. the default LDA output directory label is
# LDA_[ENTIRE_CORPUS]_[num_topics] )
OVERWRITE = False # Overwrites existing sampled comment indices. Only matters
# if ENTIRE_CORPUS = False
DOWNLOAD_RAW = True # If a raw data file is not available on disk, download it
# NOTE: Be mindful of possible changes to compression algorithm used at
# https://files.pushshift.io/reddit/comments/ beyond 02-2019, as they would
# not be reflected in the parser's code, which assumes the latest files have
# .zst extensions
CLEAN_RAW = False # After parsing, delete the raw data file from disk if it was
# not downloaded during parsing
vote_counting = True # Record the fuzzed number of upvotes when parsing
WRITE_ORIGINAL = True # Write original comments to file when parsing
author = True # Write the username of each post's author to a separate file
sentiment = True # Write sentence- and document-level sentiment of a post to
# file (based on TextBlob and Vader)
add_sentiment = True # Add CoreNLP sentiment values as a post-parsing step
# NOTE: Make sure that Stanford CoreNLP's Python package is unzipped to the
# same directory as this file and CoreNLP_server.py is also available before
# running this function.
# NOTE: Because of incompatibility with batching and hyperthreading used in
# parsing, this function should be run sequentially from NN_Book_Keeping.py
num_cores = 4 # Number of threads for sentence-by-sentence parallelization of
# CoreNLP sentiment values. Only matters if add_sentiment == True
# NOTE: Massive slow-down if the number is not slightly lower than number of cores

### Pre-processing hyperparameters

# NOTE: Matters for optimization of the parallelizations used in the functions.
# NOTE: On Brown University's supercomputer, batches of 24 months were found to
# be optimal
MaxVocab = 2000000 # maximum size of the vocabulary
FrequencyFilter = 1 # tokens with a frequency equal or less than this number
# will be filtered out of the corpus (when NN=True)
no_below = 5 # tokens that appear in less than this number of documents in
# corpus will be filtered out (when NN=False, i.e. for the LDA model)
no_above = 0.99 # tokens that appear in more than this fraction of documents in
# corpus will be filtered out
training_fraction = 0.99 # what percentage of data will be used for learning the
# LDA model. The rest of the dataset will be used as an evaluation set for
# calculating perplexity and preventing overfitting
NN_training_fraction = 0.80 # fraction of the data that is used for training
# the neural network.[1 - training_fraction] fraction of the dataset will be
# divided randomly and equally into evaluation and test sets
calculate_perc_rel = True # whether the percentage of relevant comments from
# each year should be calculated and written to file
num_process = 3 # the number of parallel processes to be executed for parsing
# NOTE: Uses Python's multiprocessing package
Neural_Relevance_Filtering = True # The dataset will be cleaned from posts
# irrelevant to the topic using a pre-trained neural network model.
# NOTE: Needs results of parsing for the same dates with WRITE_ORIGINAL==True
# NOTE: Requires a pre-trained simpletransformers model. One such model trained
# for the marijuana legalization Reddit dataset is included in the repository.
# NOTE: Default model_path is [repository path]/Human_Ratings/1_1/full_1005/
# See ROBERTA_Classifier.py for training, learning and evaluation details.
# NOTE: This task takes a long time to complete.
rel_sample_num = 200 # By default, a random sample of this size will be extracted
# from the dataset to evaluate the classification model.
balanced_rel_sample = True # whether the random filtering sample should be
# balanced across classification categories (relevant, irrelevant by default)
eval_relevance = False # F1, recall, precision and accuracy for the sample derived
# from Neural_Relevance_Filtering. Requires the sample to be complemented by
# manual labels. The default location for the sample is
# [repository path]/auto_labels/sample_auto_labeled.csv
# NOTE: Set to false if you intend to extract the relevance sample, since the produced
# files will be empty of human judgments and eval_relevance results nonsensical
num_annot = 3 # number of relevance annotators. Used to divide [rel_sample_num]
# documents evenly between the annotators with specified overlap
# NOTE: [rel_sample_num] should be divisible by this number
overlap = 0.2 # degree of overlap between annotators. Multiplying [rel_sample_num]
# by this should result in an integer

# [repository path]/original_comm/sample_auto_labeled.csv


### LDA hyperparameters
# NOTE: Number of processes for parallelization are currently set manually. See
# notes in reddit_parser.py and and Reddit_LDA_Analysis.py for more details
n_random_comments = 1500 # number of comments to sample from each year for
# training. Only matters if ENTIRE_CORPUS = False.
iterations = 1000 # number of times LDA posterior distributions will be sampled
num_threads = 5 # number of threads used for parallelized processing of comments
# Only matters if using _Threaded functions
num_topics = 100 # number of topics to be generated in each LDA sampling
alpha = 'auto' # determines how many high probability topics will be assigned to a
# document in general (not to be confused with NN l2regularization constant)
minimum_probability = 0.01 # minimum acceptable probability for an output topic
# across corpus
eta = 'auto' # determines how many high probability words will be assigned to a
# topic in general
minimum_phi_value = 0.01 # determines the lower bound on per-term topic
# probability. Only matters if per_word_topics = True.
one_hot_topic_contributions=False
# NOTE: With bad model fits sum of topic contributions for certain posts may not
# add up to close enough to 1 and the model would fail quality assurance assetion
# checks. You can examine the cases that have failed the assertion in a file
# named failures in [path], with the following format: post index, word index,
# sum of topic contributions, number of repetitions within the post
topic_cont_freq="monthly" # The frequency of topic contribution calculation
topic_idf = False # whether an inverse frequency term should be considered for
# in determining the top topics in the corpus. If set to False, contribution
# calculation will only prioritize higher overall contribution #TODO: Debug
topic_idf_thresh = 0.1 # what proportion of contributions in a post would add to
# the frequency count for a certain topic that will adversely affect its
# estimated contribution to the discourse. Only matters if topic_idf = True.
# Must be greater than zero and less than one.
# TODO: add support for a range of idf values to be tested automatically
calculate_perplexity = False # whether perplexity is calculated for the model
calculate_coherence = False # whether umass coherence is calculated for the model

### Neural Network Hyperparameters
# TODO: create a function for sampling a hundred posts at random and compare
# the neural networks on them for sentiment analysis and pro/anti based on
# unsupervised sentiment training AND supervised movie review pre-training

## determine kind of network

special_doi = False # If False, the neural network will model sentiment.
# If true, it will perform classification on comments based on a user-defined
# "dimension of interest" #TODO: Clarify "dimension of interest"
pretrained = False # whether there is sentiment analysis pre-training.
# NOTE: Should only be set to True if special_doi is also True
# NOTE: For classifier pretraining, the code should first be run with
# special_doi = False & pretrained = False and param_path should be set
# according to the output_path that results from the first run of the code
# NOTE: If pre-training is on, network hyperparameters should not be changed for
# the DOI run from the ones used for pre-training #TODO: Remove this requirement
LDA_topics = True # whether the neural networks take as part of their input
# topic contributions to each post as determined by a previously-run LDA analysis
# NOTE: the path to the LDA output to be used needs to be entered below manually
authorship = True # whether the neural networks take as part of their input
# the username of a post's author.
# NOTE: When the username is missing for posts (e.g. in case the author deleted
# their Reddit account), the model assumes a different author for each anonymous
# posting
# NOTE: The functions assume that "author" files from pre-processing are
# available in the same folder as the one containing this file
use_simple_bert = True

## Training hyperparameters
epochs = 3 # number of epochs
learning_rate = 0.003 # learning rate
#TODO: write code for automatically testing a set of learning rates
batchSz = 50 # number of parallel batches
word_embedSz = 128 # word embedding size
hiddenSz = 512 # number of units in the recurrent layer
author_embedSz = 128 # author embedding size. Matters only if authorship == True
ff1Sz = 1000 # number of units in the first feedforward layer
ff2Sz = 1000 # number of units in the second feedforward layer
keepP = 0.5 # 1 - dropout rate
early_stopping = True # whether to stop training if development set perplexity is going up
l2regularization = False # whether the model will be penalized for longer weight vectors. Helps prevent overfitting
NN_alpha = 0.01 # L2 regularization constant

### LDA-based sampling hyperparameters
# TODO: use the top topics function in Gensim to get topic-specific coherence vals
top_topic_set = list(range(num_topics)) # Choose specific topics to sample comments
# and top words
# from. set to None to use threshold or fraction of [num_topics] instead
sample_topics = 0.2 # proportion of topics that will be selected for reporting
# based on average yearly contribution. Set to None if choosing topics based on
# threshold instead.
# NOTE: Must be a valid proportion (not None) if topic_idf = True
top_topic_thresh = None # threshold for proportion contribution to the corpus
# determining topics to report. Only matters if topic_idf = False
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
num_pop = 2000 # number of the most up- or down-voted comments sampled for model
# comparison. Set to None for no sampling. Needs data parsed with
# write_original = True

### Paths

## where the data is
file_path = os.path.abspath(__file__)
model_path = os.path.dirname(file_path)
# For the neural filtering
rel_model_path = model_path+"/Human_Ratings/1_1/full_1005/"
data_path = '/users/ssloman/data/Reddit_Dataset/'
# NOTE: if not fully available on file, set Download for Parser function to
# True (source: http://files.pushshift.io/reddit/comments/)
# NOTE: if not in the same directory as this file, change the path variable
# accordingly

## Year/month combinations to get Reddit data for
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2008,2020) # year range
for year in years:
    for month in months:
        dates.append((year,month))

## where the output will be stored

# NOTE: To avoid confusion between different kinds of models, record the
# variables most important to your iteration in the folder name

## where the output will be stored

# NOTE: To avoid confusion between different kinds of models, always include doi\
# and pre in the output directory's name. After those, record the variables most
# important to your iteration

if NN: # If running a neural network analysis
    output_path = model_path+"/"+"doi_"+str(special_doi)+"_pre_"+str(pretrained)+"_e_"+str(epochs)+"_"+"hd"+"_"+str(hiddenSz)
    if not os.path.exists(output_path):
        print("Creating directory to store the output")
        os.makedirs(output_path)

    ## where the saved parameters are

    # NOTE: Enter manually. Only matters if special_doi = True and pretrained = True

    param_path = model_path+"/doi_False_pre_False_e_3_hd_512/"
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


legality_reg_expressions = []
with open("legality.txt",'r') as f:
    for line in f:
        legality_reg_expressions.append(line.lower().strip())

legality = [re.compile("|".join(legality_reg_expressions))]

# get the list of words relevant to marijuana from disk
# (requires marijuana.txt be located in the same directory)
marijuana_reg_expressions = []
with open("marijuana.txt",'r') as f:
    for line in f:
        marijuana_reg_expressions.append(line.lower().strip())

marijuana = [re.compile("|".join(marijuana_reg_expressions))]
