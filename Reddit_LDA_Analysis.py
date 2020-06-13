### import the required modules and functions
from math import ceil
import numpy as np
from config import *
from ModelEstimation import LDAModel
from reddit_parser import Parser
from Utils import *
import ast

### Create directory for storing the output if it does not already exist
if not os.path.exists(output_path):
    print("Creating directory to store the output")
    os.makedirs(output_path)

### Write hyperparameters and performance to file

Write_Performance()

### call the parsing function

# NOTE: If NN = False, will pre-process data for LDA.
# NOTE: If write_original = True, the original text of a relevant comment -
# without preprocessing - will be saved to a separate file
# NOTE: If clean_raw = True, the compressed data files will be removed from disk
# after processing
# NOTE: Relevance filters can be changed from defaults.py
# NOTE: If there is partial record on file, e.g. including some months in the
# desired range, but not all, delete the aggregated text files resulting from
# previous parsing manually before running the function again.These files can be
# identified by not having a certain month in their filenames and depending on
# parameter settings can include: author, sentiments, lda_prep, nn_prep,
# original_comm, original_indices, Random_Count_Dict, Random_Count_List,
# random_indices, RC_Count_Dict, RC_Count_List, total_count and votes

theparser = Parser()
theparser.Parse_Rel_RC_Comments()

if Neural_Relevance_Filtering:
    # Use a transformer-based neural network trained on human ratings to prune the
    # dataset from irrelevant posts. Path will default to the Human_Ratings folder
    theparser.Neural_Relevance_Screen()

    # Needs results from Neural_Relevance_Screen
    theparser.Neural_Relevance_Clean()

# Filter the dataset based on whether posts are in English (uses Google's
# language detection)
# NOTE: Requires original text of comments to be available on disk
# NOTE: Should be separately run for LDA and NN, as their preprocessed comments
# are stored in separate files
# NOTE: Performance is significantly worse for shorter posts. By default,
# the filtering is only performed on posts that contain at least 20 words
theparser.lang_filtering()
# TODO: Run the function for alternative sentiment estimates after this

## TextBlob sentiment analysis is integrated into parsing. If requested and not
# available, the following function retrieves alternative sentiment measures
# (from NLTK's Vader and CoreNLP)

# NOTE: Make sure that Stanford CoreNLP's Python package is unzipped to the
# same directory as this file and CoreNLP_server.py is also available before
# running this function.
if add_sentiment:
    theparser.add_sentiment()

## call the function for calculating the percentage of relevant comments
if calculate_perc_rel:
    theparser.Perc_Rel_RC_Comment()

### create training and evaluation sets

if not ENTIRE_CORPUS:
    theparser.select_random_comments()

## Determine the comments that will comprise each set
# NOTE: If NN = False, will create sets for LDA.
ldam = LDAModel()
ldam.Define_Sets()

## read the data and create the vocabulary and the term-document matrix
# NOTE: Needs loaded sets. Use Define_Sets() before running this function even
# if prepared sets exist on file
ldam.LDA_Corpus_Processing()

### Train and Test the LDA Model ###
ldam.get_model()

### calculate a lower bound on per-word perplexity for training and evaluation sets

# NOTE: This function writes the estimates after calculation to the file "perf"
# NOTE: This is a slow, serial function with no method for looking for previous
# estimates. Check the file named Performance in the output folder manually
# and comment out if estimates already exist

if calculate_perplexity:
    train_per_word_perplex, eval_per_word_perplex = ldam.Get_Perplexity()

### calculate umass coherence to allow for interpretability comparison between
# models with different [num_topics]

if calculate_coherence:
    ldam.Get_Coherence()

### Determine Top Topics Based on Contribution to the Model ###

# NOTE: There is a strict dependency hierarchy between the functions that come
# in this section and the next. They should be run in the order presented

### go through the corpus and calculate the contribution of each topic to comment content in each year

## Technical comments

# NOTE: The contribution is calculated over the entire dataset, not just the training set, but will ignore words not in the dictionary
# NOTE: Percentage of contributions is relative to the parts of corpus for which there WAS a reasonable prediction based on the model
# NOTE: For the LDA to give reasonable output, the number of topics given to this function should not be changed from what it was during model training
# NOTE: Serial, threaded and multicore (default) versions of this function are available (See Utils.py)
# NOTE: Even with multiprocessing, this function can be slow proportional to the number of top topics, as well as the size of the dataset

## Load or calculate topic distributions and create an enhanced version of the entire dataset
ldam.Get_Topic_Contribution()

ldam.get_top_topics()

## Plot the temporal trends in the top topics and save it to the output path
ldam.Plotter("{}/Temporal_Trend-{}-{}-{}".format(ldam.output_path,
                                                 "1hot" if one_hot_topic_contributions else "MLE", str(num_topics),
                                                 "idf" if topic_idf else "f"))

## Find the top words associated with top topics and write them to file
with open("{}/top_words-{}".format(
        output_path, "idf" if topic_idf else "f"), 'w') as f:  # create a file for storing
    # the high-probability words
    for topic in ldam.top_topics:
        print(topic, file=f)
        output = ldam.ldamodel.show_topic(topic, topn=topn)
        print(output, file=f)

### Find the most Representative Comments for the Top Topics ###
### Retrieve the probability assigned to top topics for comments in the dataset
# NOTE: This function only outputs the probabilities for comments of length at
# least [min_comm_length] with non-zero probability assigned to at least one
# top topic
ldam.Get_Top_Topic_Theta()

## call the function for sampling the most impactful comments
if num_pop != None:
    ldam.sample_pop()

### for the top topics, choose the [sample_comments] comments that reflect the
# greatest contribution of those topics and write them to file
# NOTE: If write_original was set to False during the initial parsing, this
# function will require all the original compressed data files (and will be much
# slower). If not in the same directory as this file, change the "path" argument
# NOTE: This function ignores comments in the incels subreddit, as no vote info
# for this banned subreddit exists in the source dataset
ldam.Get_Top_Comments()

## find top words associated with EVERY topic and write them to file
top_words_all = {key: [] for key in range(num_topics)}
with open(output_path + '/top_words_all_' + str(num_topics), 'a+') as f:
    # create a file for storing the high-probability words
    for topic in top_words_all.keys():
        print(topic, file=f)
        output = ldam.ldamodel.show_topic(topic, topn=topn)
        print(output, file=f)
        top_words_all[topic] = ldam.ldamodel.show_topic(topic, topn=topn)
