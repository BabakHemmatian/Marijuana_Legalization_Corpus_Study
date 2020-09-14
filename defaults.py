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

### Time interval

## Year/month combinations to get Reddit data or model results for
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2008,2020) # year range
for year in years:
    for month in months:
        dates.append((year,month))

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
add_sentiment = False # Add CoreNLP sentiment values as a post-parsing step
# NOTE: Make sure that Stanford CoreNLP's Python package is unzipped to the
# same directory as this file and CoreNLP_server.py is also available before
# running this function.
# NOTE: Because of incompatibility with batching and hyperthreading used in
# parsing, this function should be run sequentially from NN_Book_Keeping.py
num_cores = 4 # Number of threads for sentence-by-sentence parallelization of
# CoreNLP sentiment values. Only matters if add_sentiment == True
# NOTE: Slow-down if the number is not slightly lower than the number of
# physical cores on the computer

### Pre-processing hyperparameters

## Define the regex filter used for finding relevant comments during parsing

# get the list of patterns relevant to legality from disk
# (requires legality.txt in the same directory as this file)
legality_reg_expressions = []
with open("legality.txt",'r') as f:
    for line in f:
        legality_reg_expressions.append(line.lower().strip())

legality = [re.compile("|".join(legality_reg_expressions))] # compile regex

# get the list of patterns relevant to marijuana from disk
# (requires marijuana.txt in the same directory as this file)
marijuana_reg_expressions = []
with open("marijuana.txt",'r') as f:
    for line in f:
        marijuana_reg_expressions.append(line.lower().strip())

marijuana = [re.compile("|".join(marijuana_reg_expressions))] # compile regex

## Relevance filtering and evaluation hyperparameters

calculate_perc_rel = True # whether the percentage of relevant comments from
# each year should be calculated and written to file
num_process = 3 # the number of parallel processes to be executed for parsing
# NOTE: Uses Python's multiprocessing package
Neural_Relevance_Filtering = False # The dataset will be cleaned from posts
# irrelevant to the topic using a pre-trained neural network model.
# NOTE: Needs results of parsing for the same dates with WRITE_ORIGINAL==True
# NOTE: Requires a pre-trained simpletransformers model. One such model trained
# for the marijuana legalization Reddit dataset is included in the repository.
# NOTE: Default model_path is [repository path]/Human_Ratings/1_1/full_1005/
# See RoBERTa_Classifier.py for training, learning and evaluation details.
# NOTE: This task takes a long time to complete.
rel_sample_num = 200 # A random sample of this size (if available) will be
# extracted from the dataset to evaluate the classification model.
balanced_rel_sample = False # whether the random filtering sample should be
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

## Post-relevance-filtering pre-processing of the dataset

# determine the set of stopwords used in preprocessing (matters if NN=False
# or NN=True and LDA_topics = True)
keepers = ["how","should","should've","could","can","need","needn","why","few",
"more","most","all","any","against","because","ought","must","mustn","mustn't",
"shouldn","shouldn't","couldn't","couldn","shan't", "needn't"]
stop = []
for word in set(nltk.corpus.stopwords.words('english')):
    if word not in keepers:
        stop.append(str(word))
#TODO: Do we need the following for NN? How to most conveniently implement
MaxVocab = 200000 # maximum size of the LDA vocabulary
FrequencyFilter = 1 # tokens with a frequency equal or less than this number
# will be filtered out of the corpus (when NN=True)
# TODO: implement this as preprocessing for the neural network or remove. Is it
# even common anymore?
no_below = 5 # tokens that appear in less than this number of documents in
# corpus will be filtered out (when NN=False, i.e. for the LDA model)
no_above = 0.99 # tokens that appear in more than this fraction of documents in
# corpus will be filtered out (when NN=False, i.e. for the LDA model)
training_fraction = 0.90 # what percentage of data will be used for learning the
# LDA model. The rest of the dataset will be used as an evaluation set for
# calculating perplexity and identifying overfitting
NN_training_fraction = 0.90 # fraction of the data that is used for training
# the neural network. 20% of this fraction will be used for validation, while
# [1 - training_fraction] fraction will serve as a test set
# TODO: remove the need for dummy bert_prep files

### LDA hyperparameters

# NOTE: Number of processes for parallelization are currently set manually. See
# notes in reddit_parser.py and and Reddit_LDA_Analysis.py for more details
n_random_comments = 1500 # number of comments to sample from each year for
# training. Only matters if ENTIRE_CORPUS = False.
iterations = 1000 # number of times LDA posterior distributions will be sampled
num_threads = 5 # number of threads used for parallelized processing of comments
# Only matters if using _Threaded functions (OBSOLETE)
num_topics = 25 # number of topics to be generated in each LDA sampling
# NOTE: When NN=True, this variable is used to determine the results of which
# LDA model should be incorporated into the neural network classifiers if they
# are set to make use of LDA topics.
alpha = 0.1 # determines how many high probability topics will be assigned to a
# document in general (not to be confused with NN l2regularization constant)
minimum_probability = 0.01 # minimum acceptable probability for an output topic
# across corpus
eta = 0.1 # determines how many high probability words will be assigned to a
# topic in general
minimum_phi_value = 0.01 # determines the lower bound on per-term topic
# probability. Only matters if per_word_topics = True.
one_hot_topic_contributions=False
# NOTE: Determines how topic contributions are calculated. When set to True, the
# topic of each word is set to be simply the most probable topic. When False,
# the topic of each word is set to the entire probability distribution over
# num_topics topics.
# NOTE: With bad model fits sum of topic contributions for certain posts may not
# add up to close enough to 1 and the model would fail quality assurance assetion
# checks. You can examine the cases that have failed the assertion in a file
# named failures in [model_path], with the following format: post index, word index,
# sum of topic contributions, number of repetitions within the post
# TODO: check and update this. I think month and what-not was added to the info,
# or should be added if it wasn't for easier identification of the failed posts
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

# TODO: Turn rel_sample into a more generalized function for sampling a
# number of posts to compare the neural networks on them for classification of
# attitude/argumentation based on unsupervised sentiment training and across
# variations
## determine kind of network
# NOTE: If changed from a previous run, you will need to re-train your network
RoBERTa_model = "base" # whether "base" or "large" versions of default RoBERTa
# will be used
pretrained = False # whether sentiment analysis pre-training is utilized.
# NOTE: If false, RoBERTa's default weights will be used for textual context and the
# inference layers will be randomly initialized
# NOTE: If pre-training results for the particular variation of a network
# determined by NN variables below is available on disk, [param_path] should
# be set accordingly. Otherwise, if pretrained == True, sentiment pre-training
# will be started on a run of the code, which may be time-consuming. TODO: make
# sure the pre-training is included as described here
DOI = "persuasion"
# Dimension of interest for classification. Currently, "persuasion" attempts and
# "attitude" classification are defined.
LDA_topics = False # whether the neural networks take as part of their input
# topic contributions to each post as determined by a previously-run LDA analysis
# NOTE: the path to the LDA output to be used needs to be entered below manually
authorship = False # whether the neural networks take as part of their input
# the username of a post's author.
# NOTE: When the username is missing for posts (e.g. in cases where the author
# deleted their Reddit account), the model assigns the OTHER author node
# NOTE: The functions assume that "author" files from pre-processing are
# available in the same folder as the one containing this file
top_authors = 200 # the number of named authors in the dataset with the most
# posts to receive their own inference layer node if authorship = True. The rest
# will be assigned to the OTHER author node
# NOTE: the top 200 authors have at least 200 posts
use_subreddits = False # whether the inference layers learn associations with
# the more strongly represented subreddits.
top_subs = 500 # the number of subreddits in the dataset with the most posts to
# receive their own inference layer nodes if use_subreddits = True. The rest will
# be assigned to the OTHER subreddit node
# NOTE: the top 500 subreddits have at least 500 posts

## Training hyperparameters
# TODO: allow the other ones to be fed as lists too. Would simplify training
epochs = 10 # number of epochs. Should be lower for unsupervised learning (e.g.
# the sentiment pre-training) as it pertains to many more documents
learning_rate = 0.003 # learning rate
# NOTE: If fed a list of floats for training, will train iteratively using the
# various rates and outputting them to different [output_path]s. If the same
# happens when loading previously-trained models, it will load all models
# related to the various rates and performs the requested tasks on them iteratively
# for model comparison --> #TODO: write code for automatically testing a range of learning rates
batch_size = 100 # training and validation batch size
ff2Sz = 128 # number of units in the second inference layer.
# NOTE: The size of the first is determined based on the RoBERTa model used.
# The output size is determined based on the number of labels in the DOI
early_stopping = True # whether to stop training if development set perplexity is going up
# TODO: use keras to easily implement
# keepP = 0.5 # 1 - dropout rate
# l2regularization = False # whether the model will be penalized for longer weight vectors. Helps prevent overfitting
# NN_alpha = 0.01 # L2 regularization constant
# TODO: do we want l2regularization and dropout? Maybe not at first. Ask Carsten

### LDA-based sampling hyperparameters
# TODO: use the top topics function in Gensim to get topic-specific coherence vals
top_topic_set = None # Choose specific topics to sample comments and top words
# from. set to None to use threshold or fraction of [num_topics] instead below
sample_topics = 0.2 # proportion of topics that will be selected for reporting
# based on average yearly contribution. Set to None if choosing topics based on
# threshold instead.
# NOTE: Must be a valid proportion (not None) if topic_idf = True
top_topic_thresh = 0.03 # threshold for proportion contribution to the corpus
# determining topics to report. Only matters if topic_idf = False
topn = 40 # the number of high-probability words for each topic to be exported
# NOTE: Many of the words will inevitably be high probability general
# non-content and non-framing words. So topn should be set to significantly
# higher than the number of relevant words you wish to see
sample_comments = 5 # number of comments that will be sampled from top topics
min_comm_length = 40 # the minimum acceptable number of words in a sampled
# comment. Set to None for no length filtering
# NOTE: also applies to the samples extracted according to [rel_sample_num]
num_pop = None # number of the most up- or down-voted comments sampled for model
# comparison. Set to None for no sampling. Needs data parsed with
# write_original = True
# NOTE: Because of "fuzzing" to keep bots from taking advantage of the voting
# system, only the difference between up- and down-votes is extractable, not the
# absolute counts

### Paths

# NOTE: Remember to adjust the paths between local and cluster runs

# where the model is stored. Defaults to the working directory
file_path = os.path.abspath(__file__)
model_path = os.path.dirname(file_path)
# For the neural filtering
rel_model_path = model_path+"/Human_Ratings/1_1/full_1005/"
data_path = model_path + "/" # where the dataset is
# NOTE: if not fully available on file, set DOWNLOAD_RAW = True
# (source: http://files.pushshift.io/reddit/comments/)
# NOTE: if not the same directory as this file, change [data_path] accordingly

## where the output will be stored

# NOTE: To avoid confusion between different kinds of models, record the
# variables most important to your iteration in the folder name instead of the
# defaults below
# TODO: this would need to loop through different learning_rates, etc., as needed
if NN: # If running a neural network analysis
    if authorship == True:
        auth_label = top_authors
    else:
        auth_label = "False"
    output_path = "{}/{}_{}_pre_{}_lda_{}_auth_{}_subs_{}_ep_{}_rate_{}_ff2_{}".format(model_path,RoBERTa_model,DOI,pretrained,LDA_topics,auth_label,use_subreddits,epochs,learning_rate,ff2Sz)
    if not os.path.exists(output_path):
        print("Creating directory to store the output")
        os.makedirs(output_path)

    ## where the saved pre-training parameters are
    # TODO: Do we even need this? If so, name according to the pre-training needs
    param_path = output_path
    if pretrained == True:
        if not os.path.exists(param_path):
            raise Exception("Could not find saved pre-trained parameter values.")

else: # if doing topic modeling

    # Force this import so output_path is correctly set
    from lda_config import ENTIRE_CORPUS
    output_path = model_path + "/LDA_full-corpus:"+str(ENTIRE_CORPUS)+"_"+str(num_topics)
    # TODO: Correct the folder names so it doesn't train again
