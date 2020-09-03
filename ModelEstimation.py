from collections import defaultdict, OrderedDict
import csv
from functools import partial
import ast
import operator
import numpy as np
import gensim
import pickle
from math import ceil, floor
import matplotlib.pyplot as plt
import multiprocessing
import os
from pathlib2 import Path
from random import sample
import time
from config import *
from reddit_parser import Parser
import tensorflow as tf
parser_fns = Parser().get_parser_fns()
from simpletransformers.classification import ClassificationModel
from transformers import BertModel, BertTokenizer
import torch
from Utils import *


## wrapper function for calculating topic contributions to a comment
def Topic_Asgmt_Retriever_Multi_wrapper(args):
    indexed_comment, kwargs = args
    data_for_R = LDAModel(**kwargs).Topic_Asgmt_Retriever_Multi(indexed_comment)
    return data_for_R


## define the function for spawning processes to perform the calculations in parallel
# NOTE: per the recommendations of Gensim package developers, the optimal number
# of processes for the LDA modeling is (number of physical CPU cores) - 1
def theta_func(dataset, ldamodel, report):
    pool = multiprocessing.Pool(processes=3)
    func = partial(Get_LDA_Model, ldamodel=ldamodel, report=report)
    theta = pool.map(func=func, iterable=dataset)
    pool.close()
    pool.join()
    return theta


# A function that returns top topic probabilities for a given document
# (in non-zero)
def Get_LDA_Model(indexed_document, ldamodel, report):
    # get topic probabilities for the document
    topics = ldamodel.get_document_topics(ast.literal_eval(indexed_document)[1],
                                          minimum_probability=minimum_probability)

    # create a tuple including the comment index, the likely top topics and the
    # contribution of each topic to that comment if it is non-zero
    rel_probs = [(ast.literal_eval(indexed_document)[0], topic, prob) for topic, prob in topics if
                 topic in report and prob > 1e-8]

    if len(rel_probs) > 0:  # if the comment showed significant contribution of
        # at least one top topic
        return rel_probs  # return the the tuples (return None otherwise)


# Define a class of vectors in basic C that will be shared between multi-core
# prcoesses for calculating topic contribution
class Shared_Contribution_Array(object):
    ## Shared_Contribution_Array attributes
    def __init__(self, num_topics=num_topics):
        self.val = multiprocessing.RawArray('f', np.zeros([num_topics, 1]))  # shape and data type
        self.lock = multiprocessing.Lock()  # prevents different processes from writing the shared variables at the same time and mixing data up

    ## Shared_Contribution_Array update method
    def Update_Val(self, dxt):
        with self.lock:  # apply the lock
            for ind, _ in enumerate(self.val[:]):  # for each topic
                if dxt[ind, 0] != 0:  # if it was the most likely for some word in the input comment
                    self.val[ind] += dxt[ind, 0]  # add it's contribution to the yearly running sum


### Define a counter shared between multi-core processes
class Shared_Counter(object):
    ## Shared_Counter attributes
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    ## Shared_Counter incrementation method
    def Increment(self):
        with self.lock:
            self.val.value += 1

    ## Property for calling the value of the shared counter
    @property
    def value(self):
        return self.val.value


class ModelEstimator(object):
    def __init__(self, all_=ENTIRE_CORPUS, MaxVocab=MaxVocab,
                 output_path=output_path, path=model_path, dates=dates,
                 special_doi=special_doi, training_fraction=training_fraction,
                 V=OrderedDict({})):
        ## ensure the arguments have the correct types and values
        assert type(path) is str, "path variable is not an integer: %s" % str(type(path))
        assert 0 < training_fraction and 1 > training_fraction, "proportion of training set is not between zero and one"
        assert type(NN) is bool, "Modeling type variable is not a Boolean"
        # check the given path
        if not os.path.exists(path):
            raise Exception('Invalid path')
        self.all_ = all_
        self.MaxVocab = MaxVocab
        self.output_path = output_path
        self.path = path
        self.dates = dates
        self.special_doi = special_doi
        self.training_fraction = training_fraction
        self.V = V  # vocabulary

    ### function to determine comment indices for new training, development and test sets
    def Create_New_Sets(self, indices):
        print("Creating sets")

        # determine number of comments in the dataset
        if self.all_:
            if not self.special_doi:  # if not doing classification on
                # human-rated comments over a DOI
                num_comm = list(indices)[-1]  # retrieve the total number of comments
                indices = range(num_comm)  # define sets over all comments

            else:  # if doing classification on sampled comments based on a DOI
                # check to see if human comment ratings can be found on disk
                if not Path(self.fns["sample_ratings"]).is_file():
                    raise Exception("Human comment ratings for DOI training could not be found on file.")

                # TODO: Edit and test for compatibility with Qualtrics data
                # retrieve the number of comments for which there are complete human ratings
                with open(self.fns["sample_ratings"], 'r+') as csvfile:
                    reader = csv.reader(csvfile)
                    human_ratings = []  # initialize counter for the number of valid human ratings
                    # read human data for sampled comments one by one
                    for idx, row in enumerate(reader):
                        row = row[0].split(",")
                        # ignore headers and record the index of comments that are interpretable and that have ratings for all three goal variables
                        if (idx != 0 and (row[5] != 'N' or row[5] != 'n') and
                                row[2].isdigit() and row[3].isdigit() and
                                row[4].isdigit()):
                            human_ratings.append(int(row[0]))

                num_comm = len(human_ratings)  # the number of valid samples for network training
                indices = human_ratings  # define sets over sampled comments with human ratings

        else:  # if using LDA on a random subsample of the comments
            num_comm = len(indices)  # total number of sampled comments

        num_train = int(ceil(training_fraction * num_comm))  # size of training set

        if isinstance(self, NNModel):  # for NN
            num_remaining = num_comm - num_train  # the number of comments in development set or test set
            num_dev = int(floor(num_remaining / 2))  # size of the development set
            num_test = num_remaining - num_dev  # size of the test set

            self.sets['dev'] = sample(indices, num_dev)  # choose development comments at random
            remaining = set(indices).difference(self.sets['dev'])
            self.sets['test'] = sample(remaining, num_test)  # choose test comments at random
            # use the rest as training set
            self.sets['train'] = set(remaining).difference(self.sets['test'])

            # sort the indices based on position in nn_prep
            for set_key in self.set_key_list:
                self.sets[set_key] = sorted(list(self.sets[set_key]))

            # Check dev and test sets came out with right proportions
            assert (len(self.sets['dev']) - len(
                self.sets['test'])) <= 1, "The development and test set sizes are not equal"
            assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets['train']) == len(
                indices), "The sizes of the training, development and test sets do not add up to the number of posts on file"

            # write the sets to file
            for set_key in self.set_key_list:
                with open(self.path + '/' + set_key + '_set_' + str(self.special_doi), 'a+') as f:
                    for index in self.sets[set_key]:
                        print(index, end='\n', file=f)

        else:  # for LDA over the entire corpus
            num_eval = num_comm - num_train  # size of evaluation set

            self.LDA_sets['eval'] = sample(indices, num_eval)  # choose evaluation comments at random
            self.LDA_sets['train'] = set(indices).difference(
                set(self.LDA_sets['eval']))  # assign the rest of the comments to training

            # sort the indices based on position in lda_prep
            for set_key in self.LDA_set_keys:
                self.LDA_sets[set_key] = sorted(list(self.LDA_sets[set_key]))

            # Check that sets came out with right proportions
            assert len(self.LDA_sets['train']) + len(self.LDA_sets['eval']) == len(
                indices), "The training and evaluation set sizes do not correspond to the number of posts on file"

            # write the sets to file
            for set_key in self.LDA_set_keys:
                with open(self.fns["{}_set".format(set_key)], 'a+') as f:
                    for index in self.LDA_sets[set_key]:
                        print(index, end='\n', file=f)

    ### function for loading, calculating, or recalculating sets
    def Define_Sets(self):
        # load the number of comments or raise Exception if they can't be found
        findices = self.fns["counts"] if self.all_ else self.fns["random_indices"]
        try:
            assert os.path.exists(findices)
        except AssertionError:
            raise Exception("File {} not found.".format(findices))

        indices = open(findices, 'r').read().split()
        indices = filter(lambda x: x.strip(), indices)
        indices = map(int, indices)

        # if indexed comments are available (NN)
        if (isinstance(self, NNModel) and
                Path(self.fns["train_set"]).is_file() and
                Path(self.fns["dev_set"]).is_file() and
                Path(self.fns["test_set"]).is_file()):

            # determine if the comments and their relevant indices should be deleted and re-initialized or the sets should just be loaded
            Q = input("Indexed comments are already available. Do you wish to delete sets and create new ones [Y/N]?")

            # If recreating the sets is requested, delete the current ones and reinitialize
            if Q == "Y" or Q == "y":
                print("Deleting any existing sets and indexed comments")

                # delete previous record
                for set_key in self.set_key_list:
                    if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["indexed_{}_set".format(set_key)])
                    if Path(self.fns["{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["{}_set".format(set_key)])

                self.Create_New_Sets(indices)  # create sets

            # If recreating is not requested, attempt to load the sets
            elif Q == "N" or Q == "n":
                # if the sets are found, load them
                if (Path(self.fns["train_set"]).is_file()
                        and Path(self.fns["dev_set"]).is_file()
                        and Path(self.fns["test_set"]).is_file()
                ):

                    print("Loading sets from file")

                    for set_key in self.set_key_list:
                        with open(self.fns["{}_set".format(set_key)], 'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.sets[set_key].append(int(line))
                        self.sets[set_key] = np.asarray(self.sets[set_key])

                    # ensure set sizes are correct
                    assert len(self.sets['dev']) - len(
                        self.sets['test']) < 1, "The development and test set sizes are not equal"
                    assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets['train']) == len(
                        indices), "The sizes of the training, development and test sets do not add up to the number of posts on file"

                else:  # if the sets cannot be found, delete any current sets and create new sets
                    print("Failed to load previous sets. Reinitializing")

                    # delete partial record
                    for set_key in self.set_key_list:
                        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["indexed_{}_set".format(set_key)])
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    self.Create_New_Sets(indices)  # create sets

            else:  # if response was something other tha Y or N
                print("Operation aborted")
                pass

        else:  # no indexed comments available or not creating sets for NN
            # delete any possible partial indexed set
            if isinstance(self, NNModel):
                for set_key in self.set_key_list:
                    if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["indexed_{}_set".format(set_key)])

            # check to see if there are sets available, if so load them
            if (isinstance(self, NNModel) and
                Path(self.fns["train_set"]).is_file() and
                Path(self.fns["dev_set"]).is_file() and
                Path(self.fns["test_set"]).is_file()
            ) or (not isinstance(self, NNModel) and
                  Path(self.fns["train_set"]).is_file() and
                  Path(self.fns["eval_set"]).is_file()):

                print("Loading sets from file")

                if isinstance(self, NNModel):  # for NN
                    for set_key in self.set_key_list:
                        with open(self.fns["{}_set".format(set_key)], 'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.sets[set_key].append(int(line))
                        self.sets[set_key] = np.asarray(self.sets[set_key])

                    # ensure set sizes are correct
                    assert len(self.sets['dev']) - len(
                        self.sets['test']) < 1, "The sizes of the development and test sets do not match"
                    l = list(indices[-1]) if (self.all_ and not self.special_doi) else len(list(indices))
                    assert len(self.sets['dev']) + len(self.sets['test']) + len(self.sets[
                                                                                    'train']) == l, "The sizes of the training, development and test sets do not add up to the number of posts on file"

                else:  # for LDA
                    for set_key in self.LDA_set_keys:
                        with open(self.fns["{}_set".format(set_key)], 'r') as f:
                            for line in f:
                                if line.strip() != "":
                                    self.LDA_sets[set_key].append(int(line))
                        self.LDA_sets[set_key] = np.asarray(self.LDA_sets[set_key])

            else:  # if not all sets are found
                if isinstance(self, NNModel):  # for NN
                    # delete any partial set
                    for set_key in self.set_key_list:
                        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["indexed_{}_set".format(set_key)])
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    # create new sets
                    self.Create_New_Sets(indices)

                else:  # for LDA
                    # delete any partial set
                    for set_key in self.LDA_set_keys:
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    # create new sets
                    self.Create_New_Sets(indices)


class LDAModel(ModelEstimator):
    def __init__(self, alpha=alpha, corpus=None, dictionary=None, eta=eta,
                 eval_comments=None, eval_word_count=None, fns=dict(),
                 indexed_dataset=None, iterations=iterations,
                 LDA_set_keys=['train', 'eval'], LDA_sets=None, ldamodel=None,
                 min_comm_length=min_comm_length,
                 minimum_phi_value=minimum_phi_value,
                 minimum_probability=minimum_probability, no_above=no_above,
                 no_below=no_below, num_topics=num_topics,
                 one_hot=one_hot_topic_contributions, topic_idf=topic_idf,
                 topic_idf_thresh=topic_idf_thresh, stop=stop,
                 sample_comments=sample_comments, sample_topics=sample_topics,
                 top_topic_thresh=top_topic_thresh, top_topic_set=top_topic_set,
                 topic_cont_freq=topic_cont_freq, train_word_count=None,
                 **kwargs):
        ModelEstimator.__init__(self, **kwargs)
        self.alpha = alpha
        self.corpus = corpus
        self.dictionary = dictionary
        self.eta = eta
        self.eval_comments = eval_comments
        self.eval_word_count = eval_word_count
        self.indexed_dataset = indexed_dataset
        self.iterations = iterations
        self.LDA_set_keys = LDA_set_keys
        self.LDA_sets = LDA_sets
        if isinstance(self.LDA_sets, type(None)):
            self.LDA_sets = {key: [] for key in self.LDA_set_keys}
        self.ldamodel = ldamodel
        self.min_comm_length = min_comm_length
        self.minimum_phi_value = minimum_phi_value
        self.minimum_probability = minimum_probability
        self.no_above = no_above
        self.no_below = no_below
        self.num_topics = num_topics
        self.one_hot = one_hot
        self.topic_idf = topic_idf
        self.topic_idf_thresh = topic_idf_thresh
        self.sample_comments = sample_comments
        self.sample_topics = sample_topics
        self.top_topic_thresh = top_topic_thresh
        self.stop = stop
        self.topic_cont_freq = topic_cont_freq
        self.train_word_count = train_word_count
        self.fns = self.get_fns(**fns)
        self.top_topic_set = top_topic_set

    def get_fns(self, **kwargs):
        fns = {"original_comm": "{}/original_comm/original_comm".format(self.path),
               "lda_prep": "{}/lda_prep/lda_prep".format(self.path),
               "counts": parser_fns["counts"] if self.all_ else parser_fns["counts_random"],
               "indices_random": "{}/random_indices/random_indices".format(self.path),
               "train_set": "{}/LDA_train_set_{}".format(self.path, self.all_),
               "eval_set": "{}/LDA_eval_set_{}".format(self.path, self.all_),
               "corpus": "{}/RC_LDA_Corpus_{}.mm".format(self.path, self.all_),
               "eval": "{}/RC_LDA_Eval_{}.mm".format(self.path, self.all_),
               "dictionary": "{}/RC_LDA_Dict_{}.dict".format(self.path, self.all_),
               "train_word_count": "{}/train_word_count_{}".format(self.path, self.all_),
               "eval_word_count": "{}/eval_word_count_{}".format(self.path, self.all_),
               "model": "{}/RC_LDA_{}_{}.lda".format(self.path, self.num_topics, self.all_),
               "performance": "{}/Performance".format(self.output_path),
               "topic_cont": "{}/yr_topic_cont_{}-{}-{}-{}".format(self.output_path,
                                                                   "one-hot" if self.one_hot else "distributions",
                                                                   "all" if self.all_ else "subsample",
                                                                   self.topic_cont_freq,
                                                                   "idf" if self.topic_idf else "f"),
               "theta": "{}/theta_{}-{}-{}-{}".format(self.output_path,
                                                      "one-hot" if self.one_hot else "distributions",
                                                      "all" if self.all_ else "subsample", self.topic_cont_freq,
                                                      "idf" if self.topic_idf else "f"),
               "sample_keys": "{}/sample_keys-{}.csv".format(self.output_path,
                                                             "idf" if self.topic_idf else "f"),
               "sample_ratings": "{}/sample_ratings-{}.csv".format(self.output_path,
                                                                   "idf" if self.topic_idf else "f"),
               "sampled_comments": "{}/sampled_comments-{}".format(self.output_path,
                                                                   "idf" if self.topic_idf else "f"),
               "popular_comments": "{}/popular_comments-{}.csv".format(self.output_path,
                                                                       "idf" if self.topic_idf else "f"),
               "original_comm": "{}/original_comm/original_comm".format(self.path),
               "counts": "{}/counts/RC_Count_List".format(self.path),
               "votes": "{}/votes/votes".format(self.path),
               "data_for_R": "{}/data_for_R-{}.csv".format(self.output_path,
                                                           "idf" if self.topic_idf else "f")
               }
        for k, v in kwargs.items():
            fns[k] = v
        return fns

    ### calculate the yearly relevant comment counts
    def Get_Counts(self,model_path=model_path, random=False, frequency="monthly"):
        assert frequency in ("monthly", "yearly")

        fns=self.get_fns()
        fn=fns["counts"] if not random else fns["counts_random"]

        # check for monthly relevant comment counts
        if not Path(fn).is_file():
            raise Exception('The cummulative monthly counts could not be found')

        # load monthly relevant comment counts
        with open(fn,'r') as f:
            timelist = []
            for line in f:
                if line.strip() != "":
                    timelist.append(int(line))

        # intialize lists and counters
        cumulative = [] # cummulative number of comments per interval
        per = [] # number of comments per interval

        month_counter = 0

        # iterate through monthly counts
        for index,number in enumerate(timelist): # for each month
            month_counter += 1 # update counter
            if frequency=="monthly":
                cumulative.append(number) # add the cummulative count
                if index == 0: # for the first month
                    per.append(number) # append the cummulative value to number of comments per year
                else: # for the other months, subtract the last two cummulative values to find the number of relevant comments in that year
                    per.append(number - cumulative[-2])

            else:
                if (month_counter % 12) == 0 or index == len(timelist) - 1: # if at the end of the year or the corpus
                    cumulative.append(number) # add the cummulative count

                    if index + 1 == 12: # for the first year
                        per.append(number) # append the cummulative value to number of comments per year
                    else: # for the other years, subtract the last two cummulative values to find the number of relevant comments in that year
                        per.append(number - cumulative[-2])
                        month_counter = 0 # reset the counter at the end of the year

        assert sum(per) == cumulative[-1], "Monthly counts do not add up to the total count"
        assert cumulative[-1] == timelist[-1], "Total count does not add up to the number of posts on file"

        return per,cumulative

    ### Function for reading and indexing a pre-processed corpus for LDA
    def LDA_Corpus_Processing(self):
        # check the existence of pre-processed data and sets
        if not Path(self.fns["lda_prep"]).is_file():
            raise Exception('Pre-processed data could not be found')
        if (not Path(self.fns["train_set"]).is_file() or
                not Path(self.fns["eval_set"]).is_file()):
            raise Exception('Comment sets could not be found')

        # open the file storing pre-processed comments
        f = open(self.fns["lda_prep"], 'r')

        # check to see if the corpus has previously been processed
        required_files = [self.fns["corpus"], self.fns["eval"],
                          self.fns["dictionary"], self.fns["train_word_count"],
                          self.fns["eval_word_count"]]
        missing_file = 0
        for saved_file in required_files:
            if not Path(saved_file).is_file():
                missing_file += 1

        # if there is a complete extant record, load it
        if missing_file == 0:
            corpus = gensim.corpora.MmCorpus(self.fns["corpus"])
            eval_comments = gensim.corpora.MmCorpus(self.fns["eval"])
            dictionary = gensim.corpora.Dictionary.load(self.fns["dictionary"])
            with open(self.fns["train_word_count"]) as g:
                for line in g:
                    if line.strip() != "":
                        train_word_count = int(line)
            with open(self.fns["eval_word_count"]) as h:
                for line in h:
                    if line.strip() != "":
                        eval_word_count = int(line)

            print("Finished loading the dictionary and the indexed corpora from file")

        # delete any incomplete corpus record
        elif missing_file > 0 and missing_file != len(required_files):
            for saved_file in required_files:
                if Path(saved_file).is_file():
                    os.remove(saved_file)
            missing_file = len(required_files)

        # if there are no saved corpus files
        if missing_file == len(required_files):
            # timer
            print("Started processing the dataset at " + time.strftime('%l:%M%p, %m/%d/%Y'))

            f.seek(0)  # go to the beginning of the file

            # initialize a list for the corpus
            texts = []
            eval_comments = []

            train_word_count = 0  # total number of words in the filtered corpus
            eval_word_count = 0  # total number of words in the evaluation set

            ## iterate through the dataset

            for index, comment in enumerate(f):  # for each comment
                if index in self.LDA_sets['train']:  # if it belongs in the training set
                    document = []  # initialize a bag of words
                    if len(comment.strip().split()) == 1:
                        document.append(comment.strip())
                    else:
                        for word in comment.strip().split():  # for each word
                            document.append(word)

                    train_word_count += len(document)
                    texts.append(document)  # add the BOW to the corpus

                elif index in self.LDA_sets['eval']:  # if in evaluation set
                    document = []  # initialize a bag of words
                    if len(comment.strip().split()) == 1:
                        document.append(comment.strip())
                    else:
                        for word in comment.strip().split():  # for each word
                            document.append(word)

                    eval_word_count += len(document)
                    eval_comments.append(document)  # add the BOW to the corpus

                else:  # if the index is in neither set and we're processing the entire corpus, raise an Exception
                    if self.all_:
                        raise Exception('Error in processing comment index '+str(index))
                    continue

            # write the number of words in the frequency-filtered corpus to file
            with open(self.fns["train_word_count"], 'w') as u:
                print(train_word_count, file=u)

            # write the number of words in the frequency-filtered evaluation set to file
            with open(self.fns["eval_word_count"], 'w') as w:
                print(eval_word_count, file=w)

            ## create the dictionary

            dictionary = gensim.corpora.Dictionary(texts, prune_at=self.MaxVocab)  # training set
            dictionary.add_documents(eval_comments, prune_at=self.MaxVocab)  # add evaluation set
            dictionary.filter_extremes(no_below=self.no_below,
                                       no_above=self.no_above, keep_n=MaxVocab)  # filter extremes
            dictionary.save(self.fns["dictionary"])  # save dictionary to file for future use

            ## create the Bag of Words (BOW) datasets
            corpus = [dictionary.doc2bow(text) for text in texts]  # turn training comments into BOWs
            eval_comments = [dictionary.doc2bow(text) for text in eval_comments]  # turn evaluation comments into BOWs
            gensim.corpora.MmCorpus.serialize(self.fns["corpus"],
                                              corpus)  # save indexed data to file for future use (overwrites any previous versions)
            gensim.corpora.MmCorpus.serialize(self.fns["eval"], eval_comments)  # save the evaluation set to file

            # timer
            print("Finished creating the dictionary and the term-document matrices at " + time.strftime(
                '%l:%M%p, %m/%d/%Y'))

        self.dictionary = dictionary
        self.corpus = corpus
        self.eval_comments = eval_comments
        self.train_word_count = train_word_count
        self.eval_word_count = eval_word_count

    ### Train or load a trained model
    def get_model(self):
        if not Path(self.fns["model"]).is_file():  # if there are no trained models, train on the corpus
            # timer
            print("Started training LDA model at " + time.strftime('%l:%M%p, %m/%d/%Y'))

            ## create a seed for the random state generator
            seed = np.random.RandomState(0)

            ## determine the number of CPU workers for parallel processing
            # NOTE: per the recommendations of Gensim package developers, the
            # optimal number of processes for the LDA modeling is (number of
            # physical CPU cores) - 1
            workers = 3

            # define and train the LDA model
            Lda = gensim.models.ldamulticore.LdaMulticore
            self.ldamodel = Lda(self.corpus, workers=workers,
                                num_topics=self.num_topics,
                                id2word=self.dictionary,
                                iterations=self.iterations, alpha=self.alpha,
                                eta=self.eta, random_state=seed,
                                minimum_probability=self.minimum_probability,
                                per_word_topics=True,
                                minimum_phi_value=self.minimum_phi_value)
            self.ldamodel.save(self.fns["model"])  # save learned model to file for future use

            # timer
            print("Finished training model at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        else:  # if there is a trained model, load it from file
            print("Loading the trained LDA model from file")
            self.ldamodel = gensim.models.LdaMulticore.load(self.fns["model"])

    ### Get lower bounds on per-word perplexity for training and development sets (LDA)
    def Get_Perplexity(self):
        # timer
        print("Started calculating perplexity at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        ## calculate model perplexity for training and evaluation sets
        train_perplexity = self.ldamodel.bound(self.corpus,
                                               subsample_ratio=self.training_fraction)
        eval_perplexity = self.ldamodel.bound(self.eval_comments,
                                              subsample_ratio=1 - self.training_fraction)

        ## calculate per-word perplexity for training and evaluation sets
        train_per_word_perplex = np.exp2(-train_perplexity / self.train_word_count)
        eval_per_word_perplex = np.exp2(-eval_perplexity / self.eval_word_count)

        # timer
        print("Finished calculating perplexity at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        ## Print and save the per-word perplexity values to file
        with open(self.fns["performance"], 'a+') as perf:
            print("*** Perplexity ***", file=perf)
            print("Lower bound on per-word perplexity (using " + str(
                self.training_fraction) + " percent of documents as training set): " + str(train_per_word_perplex))
            print("Lower bound on per-word perplexity (using " + str(
                self.training_fraction) + " percent of documents as training set): " + str(train_per_word_perplex),
                  file=perf)
            print("Lower bound on per-word perplexity (using " + str(
                1 - self.training_fraction) + " percent of held-out documents as evaluation set): " + str(
                eval_per_word_perplex))
            print("Lower bound on per-word perplexity (using " + str(
                1 - self.training_fraction) + " percent of held-out documents as evaluation set): " + str(
                eval_per_word_perplex), file=perf)

        return train_per_word_perplex, eval_per_word_perplex

    ### Get umass coherence values for the LDA model based on training and development sets
    def Get_Coherence(self):
        # timer
        print("Started calculating coherence at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        ## calculate model coherence for training set
        umass = gensim.models.coherencemodel.CoherenceModel
        train_coherence = umass(model=self.ldamodel, corpus=self.corpus, coherence='u_mass')
        umass_train_value = train_coherence.get_coherence()

        ## Print and save coherence values to file
        with open(self.fns["performance"], 'a+') as perf:
            print("*** Coherence ***", file=perf)
            print("Umass coherence (using " + str(
                self.training_fraction) + " percent of documents as training set): " + str(umass_train_value))
            print("Umass coherence (using " + str(
                self.training_fraction) + " percent of documents as training set): " + str(umass_train_value),
                  file=perf)

        return umass_train_value

    ### function for creating an enhanced version of the dataset with year and comment indices (used in topic contribution and theta calculation)
    def Get_Indexed_Dataset(self):
        with open(self.fns["lda_prep"], 'r') as f:
            indexed_dataset = []  # initialize the full dataset

            if not self.all_:
                assert Path(
                    self.fns["random_indices"]).is_file(), "Could not find the set of chosen random indices on file"
                with open(self.fns["random_indices"]) as g:
                    rand_subsample = []
                    for line in g:
                        if line.strip() != "":
                            rand_subsample.append(int(line))

            per, cumulative = self.Get_Counts(frequency=self.topic_cont_freq)

            # Only start the counter where counts are greater than 0
            cumulative_as_arr = np.array(cumulative)
            cumulative_mask = np.ma.masked_where(cumulative_as_arr > 0, cumulative_as_arr).mask
            counter = list(cumulative_mask).index(True)
            for comm_index, comment in enumerate(f):  # for each comment
                if comm_index >= cumulative[counter]:
                    counter += 1  # update the counter if need be

                if self.all_ or (not self.all_ and comm_index in rand_subsample):
                    indexed_dataset.append((comm_index, comment, counter))
                # append the comment index, the text and the relevant month
                # or year to the dataset

        return indexed_dataset

    ### Topic Contribution (threaded) ###
    ### Define a function that retrieves the most likely topic for each word in
    # a comment and calculates
    def Topic_Asgmt_Retriever_Multi(self, indexed_comment):
        ## initialize needed vectors
        dxt = np.zeros([num_topics, 1])  # a vector for the normalized
        # contribution of each topic to the comment

        if self.topic_idf:  # if calculating inverse topic frequency
            dxf = np.zeros([num_topics, 1])  # a vector for topics that pass the
            # IDF threshold in the post

        analyzed_comment_length = 0  # a counter for the number of words in a
        # comment for which the model has predictions

        # create a list to write index, month and year, topic assignments to a
        # CSV file for analysis in R
        data_for_R = [indexed_comment[0], indexed_comment[2], []]
        if self.topic_cont_freq == "monthly":
            data_for_R.append(self.dates[0][1] + floor(float(indexed_comment[2]) / float(12)))

        comment = indexed_comment[1].strip().split()
        bow = self.dictionary.doc2bow(comment)
        # get per-word topic probabilities for the document
        gamma, phis = self.ldamodel.inference([bow], collect_sstats=True)

        for word_id, freq in bow:  # iterate over the word-topic assignments
            try:
                phi_values = [phis[i][word_id] for i in range(self.num_topics)]
            except KeyError:
                # Make sure the word either has a probability assigned to all
                # topics, or to no topics
                assert all(
                    [word_id not in phis[i] for i in range(self.num_topics)]), "Word-topic probability assignment error"
                continue

            # retrieve the phi values for various words in the comment
            topic_asgmts = sorted(enumerate(phi_values), key=lambda x: x[1],
                                  reverse=True)
            # add the most probable topic to the list to be written to a CSV
            data_for_R[2].append(topic_asgmts[0][0])

            if self.one_hot:
                dxt[topic_asgmts[0][0], 0] += freq
            else:
                assert len(
                    phi_values) == self.num_topics, "The number of phi values does not match the number of topics"
                if not Path(self.path + "/failures").is_file():
                    if not essentially_eq(sum(phi_values), freq):
                        with open(self.path + "/failures", "a+") as failures:
                            print(str(indexed_comment[0]) + "," + str(word_id) + "," + str(sum(phi_values)) + "," + str(
                                freq), file=failures)
                for topic, phi_val in enumerate(phi_values):
                    dxt[topic, 0] += phi_val
            analyzed_comment_length += freq  # update word counter

        if analyzed_comment_length > 0:  # if the model had predictions for at
            # least some of the words in the comment
            # normalize the topic contribution using comment length
            dxt = (float(1) / float(analyzed_comment_length)) * dxt

            if self.topic_idf:  # if calculating inverse topic frequency

                for topic in range(num_topics):  # for each topic
                    # if the percentage of words with predictions for which that
                    # topic is the most likely one passes [topic_idf_thresh]
                    if float(dxt[topic, 0]) / float(analyzed_comment_length) >= self.topic_idf_thresh:
                        dxf[topic, 0] += 1

                # update the shared vector of inverse topic frequency measures
                Freq_tracker[indexed_comment[2]].Update_Val(dxf)

            # update the vector of topic contributions
            Running_Sums[indexed_comment[2]].Update_Val(dxt)

        else:  # if the model had no reasonable topic proposal for any of the words in the comment
            no_predictions[indexed_comment[2]].Increment()  # update the no_predictions counter

        return data_for_R

    ### Define the main function for multi-core calculation of topic contributions
    # DEBUG: This function has been adjusted to calculate an IDF measure,
    # which hasn't been fully debugged yet. Also note that I'm basing the
    # IDF measure on only the most likely topic for each word, which is
    # maybe a fair assumption, but an assumption worth pointing out and
    # thinking about
    def Topic_Contribution_Multicore(self):
        # timer
        print("Started calculating topic contribution at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        ## check for the existence of the preprocessed dataset
        if not Path(self.fns["lda_prep"]).is_file():
            raise Exception('The preprocessed data could not be found')

        ## initialize shared vectors for yearly topic contributions
        global Running_Sums
        Running_Sums = {}
        if self.topic_idf:
            global Freq_tracker
            Freq_tracker = {}

        _, cumulative = self.Get_Counts(frequency=self.topic_cont_freq)
        per, _ = self.Get_Counts(random=not self.all_, frequency=self.topic_cont_freq)
        no_intervals = len(cumulative)

        ## Create shared counters for comments for which the model has no
        # reasonable prediction whatsoever
        global no_predictions
        no_predictions = {}

        for i in range(no_intervals):
            Running_Sums[i] = Shared_Contribution_Array(self.num_topics)
            no_predictions[i] = Shared_Counter(initval=0)
            if self.topic_idf:
                Freq_tracker[i] = Shared_Contribution_Array(self.num_topics)

        ## read and index comments
        indexed_dataset = self.Get_Indexed_Dataset()

        ## call the multiprocessing function on the dataset
        # NOTE: per the recommendations of Gensim package developers,
        # the optimal number of processes for the LDA modeling is
        # (number of physical CPU cores) - 1
        pool = multiprocessing.Pool(processes=3)
        inputs = [(indexed_comment, self.__dict__) for indexed_comment in
                  indexed_dataset]
        data_for_CSV = pool.map(func=Topic_Asgmt_Retriever_Multi_wrapper, iterable=inputs)
        pool.close()
        pool.join()

        ## Gather topic contribution estimates in one matrix
        output = []
        if self.topic_idf:
            dfs = []
        for i in range(no_intervals):
            output.append(Running_Sums[i].val[:])
            if self.topic_idf:
                dfs.append(Freq_tracker[i].val[:])

        output = np.asarray(output)
        if self.topic_idf:
            dfs = np.asarray(dfs)

        for i in range(no_intervals):
            if np.all(output[i, :] == 0):
                continue
            output[i, :] = (float(1) / (float(per[i]) - no_predictions[i].value)
                            ) * output[i, :]
            if self.topic_idf:  # if calculating inverse topic frequency
                # adjust contributions by inverse frequency in the time period
                dfs[i, :] = 1 + log(dfs[i, :] / (per[i] - no_predictions[i].value))
                # smoothen and log-transform the frequency
                output[i, :] = output[i, :] / dfs[i, :]  # adjust contribution by
                # inverse document frequency

        np.savetxt(self.fns["topic_cont"], output)  # save the topic contribution matrix to file

        ## write data_for_CSV to file
        with open(self.fns["data_for_R"], 'a+') as data:  # create the file
            writer_R = csv.writer(data)  # initialize the CSV writer
            if len(data_for_CSV[0]) == 4:  # if month information included
                writer_R.writerow(['number', 'month', 'year', 'topic_assignments'])  # write headers to the CSV file
                for comment in data_for_CSV:  # for each comment
                    month = (comment[1] + 1) % 12  # find the relevant month of year
                    if month == 0:
                        month = 12
                    writer_R.writerow([comment[0], month, comment[3],
                                       comment[2]])  # the indexing should be in line with the other output files
            elif len(data_for_CSV[0]) == 3:  # if only year info is included
                writer1.writerow(['number', 'year', 'topic_assignments'])  # write headers to the CSV file
                for comment in data_for_CSV:
                    writer_R.writerow([comment[0], comment[1], comment[2]])
            else:
                raise Exception('The topic assignments are not formatted properly.')

        # timer
        print("Finished calculating topic contributions at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        return output, indexed_dataset

    ### Function that checks for a topic contribution matrix on file and calls for its calculation if there is none
    def Get_Topic_Contribution(self):
        # check to see if topic contributions have already been calculated
        if not Path(self.fns["topic_cont"]).is_file():  # if not
            # calculate the contributions
            yr_topic_cont, indexed_dataset = self.Topic_Contribution_Multicore()
            np.savetxt(self.fns["topic_cont"], yr_topic_cont)  # save the topic contribution matrix to file

            self.indexed_dataset = indexed_dataset
            self.yr_topic_cont = yr_topic_cont

        else:  # if there are records on file
            # ask if the contributions should be loaded or calculated again
            Q = input(
                'Topic contribution estimations were found on file. Do you wish to delete them and calculate contributions again? [Y/N]')

            if Q == 'Y' or Q == 'y':  # re-calculate
                # calculate the contributions
                yr_topic_cont, indexed_dataset = self.Topic_Contribution_Multicore()
                np.savetxt(self.fns["topic_cont"], yr_topic_cont)  # save the topic contribution matrix to file

                self.indexed_dataset = indexed_dataset
                self.yr_topic_cont = yr_topic_cont

            elif Q == 'N' or Q == 'n':  # load from file
                print("Loading topic contributions and indexed dataset from file")

                indexed_dataset = self.Get_Indexed_Dataset()
                yr_topic_cont = np.loadtxt(self.fns["topic_cont"])

                self.indexed_dataset = indexed_dataset
                self.yr_topic_cont = yr_topic_cont

            else:  # if the answer is neither yes, nor no
                print(
                    "Operation aborted. Please note that loaded topic contribution matrix and indexed dataset are required for determining top topics and sampling comments.")
                pass

    # Determine the top topics based on average per-comment contribution over time
    def get_top_topics(self, yr_topic_cont=None):
        if isinstance(yr_topic_cont, type(None)):
            yr_topic_cont = self.yr_topic_cont
        # initialize a vector for average topic contribution

        # get the comment count for each month
        per, cumulative = self.Get_Counts(frequency=self.topic_cont_freq)

        # scale contribution based on the number of comments in each month
        scaled = np.zeros_like(yr_topic_cont)
        for i in range(len(per)):
            if per[i] != 0:
                scaled[i, :] = (float(per[i]) / float(cumulative[-1])) * yr_topic_cont[i, :]

        # average contribution for each topic
        avg_cont = np.empty(self.num_topics)
        for i in range(self.num_topics):
            # avg_cont[i] = np.sum(scaled[:,i])
            avg_cont[i] = np.mean(yr_topic_cont[:, i])

        # Find the indices of the [sample_topics] fraction of topics that have
        # the greatest average contribution to the model
        if not isinstance(sample_topics, type(None)):  # if based on fraction
            top_topic_no = int(ceil(self.sample_topics * self.num_topics))
            self.top_topics = avg_cont.argsort()[-top_topic_no:][::-1]
        # or the indices of topics that have passed the threshold for
        # consideration based on average contribution
        else:  # if based on threshold
            self.top_topics = np.where(avg_cont >= top_topic_thresh)[0]

        # If a set of top topics is provided beforehand
        if not isinstance(self.top_topic_set, type(None)):
            self.top_topics = self.top_topic_set

        print(self.top_topics)
        with open("{}/top_topic_ids-{}".format(self.output_path, "idf" if self.topic_idf else "f"), "w") as f:
            for topic in self.top_topics:
                print(topic, end="\n", file=f)

    ### Define a function for plotting the temporal trends in the top topics
    def Plotter(self, name):
        plotter = []
        for topic in self.top_topics:
            plotter.append(self.yr_topic_cont[:, topic].tolist())

        plots = {}
        for i in range(len(self.top_topics.tolist())):
            plots[i] = plt.plot(range(1, len(plotter[0]) + 1), plotter[i],
                                label='Topic ' + str(self.top_topics[i]))
        plt.legend(loc='best')
        plt.xlabel('{}/{}-{}/{}'.format(dates[0][1], dates[0][0], dates[-1][1],
                                        dates[-1][0]))
        plt.ylabel('Topic Probability')
        plt.title('Contribution of the top topics to the LDA model for {}/{}-{}/{}'.format(
            dates[0][1], dates[0][0], dates[-1][1], dates[-1][0]))
        plt.grid(True)
        plt.savefig(name)

    ### Function for multi-core processing of comment-top topic probabilities

    # NOTE: The current version is memory-intensive. For a dataset of ~2mil
    # posts, ~3GB of RAM was needed.

    ### TODO: Add functionality for choosing a certain year (or interval) for
    # which we ask the program to sample comments (use indexed_dataset[2])
    def Top_Topics_Theta_Multicore(self):
        # timer
        print("Started calculating theta at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        if not Path(self.output_path + "/filtered_dataset").is_file():
            with open(self.output_path + "/filtered_dataset", "a+") as filtered_dataset:  # initialize dataset
                for document in self.indexed_dataset:  # for each comment in the
                    # indexed_dataset
                    if self.min_comm_length == None:  # if not filtering based on comment
                        # length, add a tuple including comment index, bag of words
                        # representation and relevant time interval to the dataset
                        print(str((document[0],
                                   self.dictionary.doc2bow(document[1].strip().split()),
                                   document[2])), file=filtered_dataset)

                    else:  # if filtering based on comment length
                        if len(document[1].strip().split()) > self.min_comm_length:
                            # filter out short comments
                            # add a tuple including comment index, bag of words
                            # representation and relevant time interval to the dataset
                            print(str((document[0],
                                       self.dictionary.doc2bow(document[1].strip().split()),
                                       document[2])), file=filtered_dataset)

        dataset = open(self.output_path + "/filtered_dataset", "r")

        ## call the multiprocessing function on the dataset
        theta_with_none = theta_func(dataset, self.ldamodel, self.top_topics)

        ## flatten the list and get rid of 'None's
        theta = []
        for comment in theta_with_none:
            if comment is not None:
                for item in comment:
                    theta.append(item)

        # timer
        print("Finished calculating theta at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        return theta

    ### Function that calls for calculating, re-calculating or loading theta
    # estimations for top topics
    def Get_Top_Topic_Theta(self):
        # check to see if theta for top topics has already been calculated
        if not Path(self.fns["theta"]).is_file():  # if not
            theta = self.Top_Topics_Theta_Multicore()  # calculate theta

            # save theta to file
            with open(self.fns["theta"], 'a+') as f:
                for element in theta:
                    f.write(' '.join(str(number) for number in element) + '\n')

            self.theta = theta

        else:  # if there are records on file
            # ask if theta should be loaded or calculated again
            Q = input(
                'Theta estimations were found on file. Do you wish to delete them and calculate probabilities again? [Y/N]')

            if Q == 'Y' or Q == 'y':  # re-calculate
                os.remove(self.fns["theta"])  # delete the old records

                theta = self.Top_Topics_Theta_Multicore()  # calculate theta

                # save theta to file
                with open(self.fns["theta"], 'a+') as f:
                    for element in theta:
                        f.write(' '.join(str(number) for number in element) + '\n')

                self.theta = theta

            elif Q == 'N' or Q == 'n':  # load from file
                print("Loading theta from file")

                with open(self.fns["theta"], 'r') as f:
                    theta = [tuple(map(float, number.split())) for number in f]

                self.theta = theta

            else:  # if the answer is neither yes, nor no
                print("Operation aborted. Please note that loaded theta is required for sampling top comments.")

    ### Defines a function for finding the [sample_comments] most representative
    # length-filtered comments for each top topic
    def Top_Comment_Indices(self):
        top_topic_probs = {}  # initialize a dictionary for all top comment idx
        sampled_indices = {}  # initialize a dictionary for storing sampled
        # comment idx
        sampled_probs = {}  # initialize a list for storing top topic
        # contribution to sampled comments
        sampled_ids = {}  # intialize a dict for storing randomly chosen
        # six-digit IDs for sampled comments

        for topic in self.top_topics:  # for each top topic
            # find all comments with significant contribution from that topic
            top_topic_probs[topic] = [element for element in self.theta if
                                      element[1] == topic]
            top_topic_probs[topic] = sorted(top_topic_probs[topic], key=lambda x: x[2],
                                            reverse=True)  # sort them based on topic contribution

            # find the [sample_comments] comments for each top topic that show the greatest contribution
            sampled_indices[topic] = []
            sampled_probs[topic] = []

            sampled_counter = 0
            for element in top_topic_probs[topic]:
                repeated = 0
                if len(sampled_probs[topic]) != 0:
                    for other_element in sampled_probs[topic]:
                        if essentially_eq(element[2],other_element):
                            repeated = 1

                if repeated == 0:
                    sampled_indices[topic].append(element[0])  # record the index
                    sampled_probs[topic].append(element[2])  # record the contribution of the topic
                    # suggest a random 8-digit id for the sampled comment
                    prop_id = np.random.random_integers(low=10000000, high=99999999)
                    while prop_id in sampled_ids:  # resample in the unlikely event the id is already in use
                        prop_id = np.random.random_integers(low=10000000, high=99999999)
                    sampled_ids[element[0]] = prop_id  # store the random id
                    sampled_counter += 1
                    if sampled_counter == self.sample_comments:
                        break

        return sampled_indices, sampled_probs, sampled_ids

    ### retrieve the original text of sampled comments and write them to file
    # along with the relevant topic ID
    # TODO: Should add the possibility of sampling from specific year(s)
    def Get_Top_Comments(self):
        # timer
        print("Started sampling top comments at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        # find the top comments associated with each top topic
        sampled_indices, sampled_probs, sampled_ids = self.Top_Comment_Indices()

        _, yearly_cumulative = self.Get_Counts(frequency="yearly")
        _, monthly_cumulative = self.Get_Counts(frequency="monthly")

        if not Path(self.fns[
                        "original_comm"]).is_file():  # if the original relevant comments are not already available on disk, read them from the original compressed files
            # json parser
            decoder = json.JSONDecoder(encoding='utf-8')

            # check for the presence of data files
            if not glob.glob(self.path + '/*.bz2'):
                raise Exception('No data file found')

            # open two CSV files for recording sampled comment keys and ratings
            with open(self.fns["sample_keys"], 'a+') as sample_keys, \
                    open(self.fns["sample_ratings"], 'a+') as sample_ratings, \
                    open(self.fns["sampled_comments"], 'a+') as fout:  # determine the I/O files:

                ## iterate over files in directory to find the relevant documents
                counter = 0  # counting the number of all processed comments
                year_counter = 0  # the first year in the corpus

                # create CSV file for recording sampled comment information
                writer_keys = csv.writer(sample_keys)  # initialize the CSV writer
                writer_keys.writerow(['comment number', 'random index', 'month', 'year', 'topic',
                                      'contribution'])  # write headers to the CSV file

                # create CSV file for recording sampled comment ratings
                writer_ratings = csv.writer(sample_ratings)
                writer_ratings.writerow(['index', 'pro', 'values', 'consequences', 'preferences', 'interpretability'])

                bag_of_comments = {}  # initialize a list for the sampled
                # original comments. Will be used to shuffle comments before
                # writing them to file

                # iterate through the files in the 'path' directory in alphabetic order
                # TODO: Make it independent of having all the files on disk
                for filename in sorted(os.listdir(self.path)):
                    # only include relevant files
                    if os.path.splitext(filename)[1] == '.bz2' and 'RC_' in filename:
                        ## prepare files
                        # open the file as a text file, in utf8 encoding
                        with bz2.BZ2File(filename, 'r') as fin:

                            ## read data
                            for line in fin:  # for each comment
                                # parse the json, and turn it into regular text
                                comment = decoder.decode(line)
                                original_body = HTMLParser.HTMLParser().unescape(
                                    comment["body"])  # remove HTML characters

                                # filter comments by relevance to the topic
                                if len(GAYMAR.findall(original_body)) > 0 or len(MAREQU.findall(original_body)) > 0:
                                    # clean the text for LDA
                                    body = Parser(stop=self.stop).LDA_clean(original_body)

                                    # if the comment body is not empty after preprocessing
                                    if body.strip() != "":
                                        counter += 1  # update the counter

                                        # update year counter if need be
                                        if counter - 1 >= yearly_cumulative[year_counter]:
                                            year_counter += 1

                                        # find the relevant month
                                        for value in monthly_cumulative:
                                            if value > counter - 1:
                                                month = (monthly_cumulative.index(value) + 1) % 12
                                                if month == 0:
                                                    month = 12
                                                break

                                        for topic, indices in sampled_indices.items():
                                            if counter - 1 in indices:
                                                # remove mid-comment lines and set encoding
                                                original_body = original_body.replace("\n", "")
                                                original_body = original_body.encode("utf-8")

                                                bag_of_comments[sampled_ids[counter - 1]] = " ".join(
                                                    comment.strip().split())

                                                # print the values to CSV file
                                                itemindex = sampled_indices[topic].index(comm_index)
                                                writer_keys.writerow([counter - 1, sampled_ids[counter - 1], month,
                                                                      self.dates[0][1] + year_counter, topic,
                                                                      sampled_probs[topic][itemindex]])

                                                # print the comment to file
                                                print(" ".join(original_body.strip().split()), file=fout)

                                                break  # if you found the index in one of the topics, no reason to keep looking

                ## shuffle the comments and write to file for human raters
                random_keys = np.random.permutation(list(bag_of_comments.keys()))
                for random_key in random_keys:
                    writer_ratings.writerow([random_key])
                    print('index: ' + str(random_key), file=fout)
                    print(" ".join(bag_of_comments[random_key].strip().split()), file=fout)

            # timer
            print("Finished sampling top comments at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        else:  # if a file containing only the original relevant comments is available on disk
            with open(self.fns["original_comm"], 'r') as fin, \
                    open(self.fns["sample_keys"], 'a+') as sample_keys, \
                    open(self.fns["sample_ratings"], 'a+') as sample_ratings, \
                    open(self.fns["sampled_comments"], 'a+') as fout:  # determine the I/O files

                # create CSV file for recording sampled comment information
                writer_keys = csv.writer(sample_keys)  # initialize the CSV writer
                writer_keys.writerow(['comment number', 'random index', 'month', 'year', 'topic',
                                      'contribution'])  # write headers to the CSV file

                # create CSV file for recording sampled comment ratings
                writer_ratings = csv.writer(sample_ratings)
                writer_ratings.writerow(['index', 'pro', 'values', 'consequences', 'preferences', 'interpretability'])

                bag_of_comments = {}  # initialize a list for the sampled
                # original comments. Will be used to shuffle comments before
                # writing them to file
                year_counter = 0  # initialize a counter for the comment's year

                for comm_index, comment in enumerate(fin):  # iterate over the original comments

                    # update year counter if need be
                    if comm_index >= yearly_cumulative[year_counter]:
                        year_counter += 1

                    for topic, indices in sampled_indices.items():
                        if comm_index in indices:

                            # find the relevant month
                            for value in monthly_cumulative:
                                if value > comm_index:
                                    month = (monthly_cumulative.index(value) + 1) % 12
                                    if month == 0:
                                        month = 12
                                    break

                            bag_of_comments[sampled_ids[comm_index]] = " ".join(comment.strip().split())

                            # print the values to CSV file
                            itemindex = sampled_indices[topic].index(comm_index)
                            writer_keys.writerow(
                                [comm_index, sampled_ids[comm_index], month, self.dates[0][1] + year_counter, topic,
                                 sampled_probs[topic][itemindex]])

                            break  # if you found the index in one of the topics, no reason to keep looking

                ## shuffle the comments and write to file for human raters
                random_keys = np.random.permutation(list(bag_of_comments.keys()))
                for random_key in random_keys:
                    writer_ratings.writerow([random_key])
                    print('index: ' + str(random_key), file=fout)
                    print(" ".join(bag_of_comments[random_key].strip().split()), file=fout)

                # timer
                print("Finished sampling top comments at " + time.strftime('%l:%M%p, %m/%d/%Y'))

    ## function for sampling the most impactful comments
    def sample_pop(self, num_pop=num_pop, min_comm_length=min_comm_length):

        assert Path(self.path + "counts/RC_Count_List").is_file()
        assert Path(self.path + "original_comm/original_comm").is_file()
        assert Path(self.path + "votes/votes").is_file()

        # Retrieve the list of upvote/downvote values
        with open(self.path + "counts/RC_Count_List", 'r') as f:
            timelist = []
            for line in f:
                if line.strip() != "":
                    timelist.append(int(line))

        # Retrieve the text of the original comments
        with open(self.path + "original_comm/original_comm", 'r') as f:
            orig_comm = []
            for line in f:
                if line.strip() != "":
                    orig_comm.append(line.strip())

        # Retrieve the pre-processed comments
        with open("lda_prep", 'r') as f:
            lda_prep = []
            for line in f:
                if line.strip() != "":
                    lda_prep.append(line.strip())

        # Calculate the relevant month and year for each comment
        rel_month = np.zeros((timelist[-1], 1), dtype="int32")
        month_of_year = np.zeros((timelist[-1], 1), dtype="int32")
        rel_year = np.zeros((timelist[-1], 1), dtype="int32")
        for rel_ind in range(timelist[-1]):
            # Find the relevant month and year
            rel_month[rel_ind, 0] = next((i + 1 for i in range(len(timelist)) if timelist[i] > rel_ind), 141)
            month_of_year[rel_ind, 0] = rel_month[rel_ind, 0] % 12
            if month_of_year[rel_ind, 0] == 0:
                month_of_year[rel_ind, 0] = 12
            rel_year[rel_ind, 0] = int(self.dates[0][1] + int(floor(rel_month[rel_ind, 0] / 12)))

        # retrieve comment scores from file
        with open("votes", 'r') as f:
            vote_count = dict()
            abs_vote_count = dict()
            for number, line in enumerate(f):
                if line.strip() != "" and line.strip() != "None":
                    vote_count[str(number)] = int(line)
                    abs_vote_count[str(number)] = abs(int(line))

        # sort scores based on absolute value
        sorted_votes = sorted(abs_vote_count.items(), key=operator.itemgetter(1), reverse=True)

        # Find the num_pop comments with the highest impact on the discourse
        counter = 0
        results = []
        abs_results = []
        popular = []
        for x in sorted_votes:
            comment = orig_comm[int(x[0])]
            if min_comm_length != None:
                if len(comment.strip().split()) > min_comm_length:
                    counter += 1
                    results.append(vote_count[x[0]])
                    abs_results.append(x[1])
                    popular.append(int(x[0]))
            else:
                counter += 1
                results.append(vote_count[x[0]])
                abs_results.append(x[1])
                popular.append(int(x[0]))
            if counter == num_pop:
                break

        # retrieve the text of popular comments
        pop_orig_texts = []
        pop_proc_texts = []
        for pop_comment in popular:
            pop_orig_texts.append(orig_comm[pop_comment])
        for pop_comment in popular:
            pop_proc_texts.append(lda_prep[pop_comment])

        # Retrieve topic probability from the model for each popular comment
        pop_comm_topic = []
        pop_comm_contrib = []
        for comment in pop_proc_texts:
            comment = comment.strip().split()
            bow = self.dictionary.doc2bow(comment)
            # get topic probabilities for the document
            topic_dist = self.ldamodel.get_document_topics([bow], per_word_topics=False)
            # sort the probabilities
            sorted_topics = sorted(topic_dist[0], key=lambda x: x[1], reverse=True)
            # add the most probable topic to the list to be written to a CSV
            # pop_comm_topic.append(sorted_topics[0][0])
            # pop_comm_contrib.append(sorted_topics[0][1])
            pop_comm_topic.append(sorted_topics)  # this appends ALL contributing topics

        # write the most popular comments, their timeframe and score to file
        with open(self.fns["popular_comments"], 'w+') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['text','year','month','score','topic','contribution'])
            writer.writerow(['text', 'year', 'month', 'score', 'topic contribution'])
            for number, pop_comment in enumerate(popular):
                # writer.writerow([pop_orig_texts[number],rel_year[pop_comment,0],
                # month_of_year[pop_comment,0],results[number],
                # pop_comm_topic[number],pop_comm_contrib[number]])
                writer.writerow([pop_orig_texts[number], rel_year[pop_comment, 0],
                                 month_of_year[pop_comment, 0], results[number],
                                 pop_comm_topic[number]])


class NNModel(ModelEstimator):
    def __init__(self, use_simple_bert=use_simple_bert, FrequencyFilter=FrequencyFilter, learning_rate=learning_rate,
                 batchSz=batchSz, word_embedSz=word_embedSz, hiddenSz=hiddenSz,
                 author_embedSz=author_embedSz, ff1Sz=ff1Sz, ff2Sz=ff2Sz,
                 keepP=keepP, l2regularization=l2regularization, NN_alpha=NN_alpha,
                 early_stopping=early_stopping, LDA_topics=LDA_topics, num_topics=num_topics,
                 authorship=authorship, **kwargs):
        ModelEstimator.__init__(self, **kwargs)
        self.FrequencyFilter = FrequencyFilter
        self.learning_rate = learning_rate
        self.batchSz = batchSz
        self.author_embedSz=author_embedSz
        self.word_embedSz = word_embedSz
        self.hiddenSz = hiddenSz
        self.ff1Sz = ff1Sz
        self.ff2Sz = ff2Sz
        self.keepP = keepP
        self.l2regularization = l2regularization
        if self.l2regularization:
            self.NN_alpha = NN_alpha
        self.early_stopping = early_stopping
        self.LDA_topics = LDA_topics
        self.num_topics = num_topics
        self.authorship = authorship
        self.set_key_list = ['train', 'dev', 'test']  # for NN
        self.sets = {key: [] for key in self.set_key_list}  # for NN
        self.indices = {key: [] for key in self.set_key_list}
        self.lengths = {key: [] for key in self.set_key_list}
        self.Max = {key: [] for key in self.set_key_list}
        self.Max_l = None
        self.sentiments = {key: [] for key in self.set_key_list}  # for NN
        self.accuracy = {key: [] for key in self.set_key_list}
        for set_key in self.set_key_list:
            self.accuracy[set_key] = np.empty(epochs)

        self.fns = self.get_fns()

    def train_bert_model(self, train_data):
        if self.use_simple_bert:
            self.bert_model = ClassificationModel('roberta', 'roberta-base',
                                              args={"output_hidden_states" : True})  # Model for word embeddings

        else:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

       # self.bert_model.train_model(train_data)
    def bert_predictions(self, sentence):
        if use_simple_bert:
            to_predict = [sentence]
            _, _, all_embedding_outputs, hidden_states = self.bert_model.predict(to_predict)
            # have a function reading in from the 3 files, average and then determine the correct label -- then attach the correct label to each document
            # use author indices to pass in as additional args to bert
            return hidden_states
        else:
            input_ids = torch.tensor(self.bert_tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = self.bert_model(input_ids)
            last_hidden_states = outputs[0]
            return last_hidden_states

    def get_fns(self, **kwargs):
        fns = {"nn_prep": "{}/nn_prep".format(self.path),
               "counts": parser_fns["counts"] if self.all_ else parser_fns["counts_random"],
               "dictionary": "{}/dict_{}".format(self.path, self.special_doi),
               "train_set": "{}/train_{}".format(self.path, self.special_doi),
               "dev_set": "{}/dev_{}".format(self.path, self.special_doi),
               "test_set": "{}/test_{}".format(self.path, self.special_doi),
               "indexed_train_set": "{}/indexed_train_{}".format(self.path, self.special_doi),
               "indexed_dev_set": "{}/indexed_dev_{}".format(self.path, self.special_doi),
               "indexed_test_set": "{}/indexed_test_{}".format(self.path, self.special_doi),
               "sample_ratings": "{}/sample_ratings.csv".format(self.output_path),
               "author": "{}/author".format(self.path),
               "sentiments": "{}/sentiments".format(self.path)
               }
        for k, v in kwargs.items():
            fns[k] = v
        return fns

    ### load or create vocabulary and load or create indexed versions of comments in sets
    # NOTE: Only for NN. For LDA we use gensim's dictionary functions
    def Index_Set(self, set_key):
        ## record word frequency in the entire dataset
        frequency = defaultdict(int)
        if Path(self.fns["nn_prep"]).is_file():  # look for preprocessed data
            fin = open(self.fns["nn_prep"], 'r')
            for comment in fin:  # for each comment
                for token in comment.split():  # for each word
                    frequency[token] += 1  # count the number of occurrences

        else:  # if no data is found, raise an error
            raise Exception('Pre-processed dataset could not be found')

        # if indexed comments are available and we are trying to index the training set
        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file() and set_key == 'train':
            # If the vocabulary is available, load it
            if Path(self.path + "/dict_" + str(self.special_doi)).is_file():
                print("Loading dictionary from file")

                with open(self.fns["dictionary"], 'r') as f:
                    for line in f:
                        if line.strip() != "":
                            (key, val) = line.split()
                            V[key] = int(val)

            else:  # if the vocabulary is not available
                # delete the possible dictionary-less indexed training set file
                if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                    os.remove(self.fns["indexed_{}_set".format(set_key)])

        # if indexed comments are available, load them
        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
            print("Loading the set from file")

            with open(self.fns["indexed_{}_set".format(set_key)], 'r') as f:
                for line in f:
                    assert line.strip() != ""
                    comment = []
                    for index in line.split():
                        comment.append(index)
                    self.indices[set_key].append(comment)

        else:  # if the indexed comments are not available, create them
            if set_key == 'train':  # for training set
                # timer
                print("Started creating the dictionary at " + time.strftime('%l:%M%p, %m/%d/%Y'))

                ## initialize the vocabulary with various UNKs
                self.V.update(
                    {"*STOP2*": 1, "*UNK*": 2, "*UNKED*": 3, "*UNKS*": 4, "*UNKING*": 5, "*UNKLY*": 6, "*UNKER*": 7,
                     "*UNKION*": 8, "*UNKAL*": 9, "*UNKOUS*": 10, "*STOP*": 11})

            ## read the dataset and index the relevant comments
            fin.seek(0)  # go to the beginning of the data file
            for counter, comm in enumerate(fin):  # for each comment
                if counter in self.sets[set_key]:  # if it belongs in the set
                    comment = []  # initialize a list

                    for word in comm.split():  # for each word
                        if frequency[word] > FrequencyFilter:  # filter non-frequent words
                            if word in self.V.keys():  # if the word is already in the vocabulary
                                comment.append(self.V[word])  # index it and add it to the list

                            elif set_key == 'train':  # if the word is not in vocabulary and we are indexing the training set
                                if len(
                                        self.V) - 11 <= self.MaxVocab:  # if the vocabulary still has room (not counting STOPs and UNKs)
                                    self.V[word] = len(self.V) + 1  # give it an index (leave index 0 for padding)
                                    comment.append(self.V[word])  # append it to the list of words

                                else:  # if the vocabulary doesn't have room, assign the word to an UNK according to its suffix or lack thereof
                                    if word.endswith("ed"):
                                        comment.append(3)
                                    elif word.endswith("s"):
                                        comment.append(4)
                                    elif word.endswith("ing"):
                                        comment.append(5)
                                    elif word.endswith("ly"):
                                        comment.append(6)
                                    elif word.endswith("er"):
                                        comment.append(7)
                                    elif word.endswith("ion"):
                                        comment.append(8)
                                    elif word.endswith("al"):
                                        comment.append(9)
                                    elif word.endswith("ous"):
                                        comment.append(10)

                                    else:  # if the word doesn't have any easily identifiable suffix
                                        comment.append(2)

                            else:  # the word is not in vocabulary and we are not indexing the training set
                                if word.endswith("ed"):
                                    comment.append(3)
                                elif word.endswith("s"):
                                    comment.append(4)
                                elif word.endswith("ing"):
                                    comment.append(5)
                                elif word.endswith("ly"):
                                    comment.append(6)
                                elif word.endswith("er"):
                                    comment.append(7)
                                elif word.endswith("ion"):
                                    comment.append(8)
                                elif word.endswith("al"):
                                    comment.append(9)
                                elif word.endswith("ous"):
                                    comment.append(10)

                                else:  # if the word doesn't have any easily identifiable suffix
                                    comment.append(2)

                    self.indices[set_key].append(comment)  # add the comment to the indexed list

            ## save the vocabulary to file
            if set_key == 'train':
                vocab = open(self.fns["dictionary"], 'a+')
                for word, index in self.V.items():
                    print(word + " " + str(index), file=vocab)
                vocab.close

            ## save the indexed datasets to file
            with open(self.fns["indexed_{}_set".format(set_key)], 'a+') as f:
                for comment in self.indices[set_key]:
                    assert len(comment) != 0
                    for ind, word in enumerate(comment):
                        if ind != len(comment) - 1:
                            print(word, end=" ", file=f)
                        elif ind == len(comment) - 1:
                            print(word, file=f)

            # ensure that datasets have the right size
            assert len(self.indices[set_key]) == len(self.sets[set_key])

        # timer
        print("Finished indexing the " + set_key + " set at " + time.strftime('%l:%M%p, %m/%d/%Y'))

    ## function for getting average sentiment values for training, development
    # and test sets
    def Get_Sentiment(self, path):
        if not Path(self.fns["sentiments"]).is_file():
            raise Exception("Sentiment data could not be found on file.")
        else:
            with open(self.fns["sentiments"], "r") as sentiments:
                for idx, sentiment in enumerate(sentiments):
                    if idx in self.sets["train"]:
                        self.sentiments["train"].append(sentiment)
                    elif idx in self.sets["dev"]:
                        self.sentiments["dev"].append(sentiment)
                    elif idx in self.sets["test"]:
                        self.sentiments["test"].append(sentiment)
                    else:
                        raise Exception("The set indices are not correctly defined")

    def NN_param_typecheck(self):
        assert 0 < self.learning_rate and 1 > self.learning_rate, "invalid learning rate"

        assert type(self.batchSz) is int, "invalid batch size"
        assert type(self.word_embedSz) is int, "invalid word embedding size"
        assert type(self.hiddenSz) is int, "invalid hidden layer size"
        if self.authorship:
            assert type(self.author_embedSz) is int, "invalid authorship embedding size"
        assert type(self.ff1Sz) is int, "invalid feedforward layer size"
        assert type(self.ff2Sz) is int, "invalid feedforward layer size"
        assert 0 < self.keepP and 1 >= self.keepP, "invalid dropout rate"
        assert type(self.l2regularization) is bool, "invalid regularization parameter"
        if self.l2regularization == True:
            assert 0 < self.alpha and 1 > self.alpha, "invalid alpha parameter for regularization"
        assert type(self.early_stopping) is bool, "invalid early_stopping parameter"

    ## Function for creating the neural network's computation graph

    def Setup_Comp_Graph(self, device_count=None):

        if not device_count is None:
            self.device_count = device_count

        ## create placeholders for input, output, loss weights and dropout rate

        inpt = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        if self.LDA_topics:
            lda_inpt = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
        if self.authorship:
            authorship_inpt = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
        answr = tf.compat.v1.placeholder(tf.int32, shape=[None, None])
        DOutRate = tf.compat.v1.placeholder(tf.float32)

        ## set up the graph parameters

        # for pre-trained classification network, load parameters from file

        if self.special_doi and self.pretrained:
            print("Loading parameter estimates from file")
            word_embed = np.loadtxt(param_path + "word_embed", dtype='float32')
            if self.authorship:
                author_embed = np.loadtxt(param_path + "authorship_layer", dtype='float32')
            pre_state = np.loadtxt(param_path + "state", dtype='float32')
            weights1 = np.loadtxt(param_path + "weights1", dtype='float32')
            biases1 = np.loadtxt(param_path + "biases1", dtype='float32')
            weights2 = np.loadtxt(param_path + "weights2", dtype='float32')
            biases2 = np.loadtxt(param_path + "biases2", dtype='float32')
            weights3 = np.loadtxt(param_path + "weights3", dtype='float32')
            biases3 = np.loadtxt(param_path + "biases3", dtype='float32')
        else:
            print("Initializing parameter estimates")

        # initial word embeddings

        if self.special_doi and self.pretrained:
            E = tf.Variable(word_embed)
        else:
            E = tf.Variable(tf.random.normal([len(V), word_embedSz], stddev=0.1))

        # look up the embeddings
        embed = tf.nn.embedding_lookup(params=E, ids=inpt)

        # calculate sum of the weights for l2regularization
        if l2regularization == True:
            sum_weights = tf.nn.l2_loss(embed)

        # define the recurrent layer (Gated Recurrent Unit)
        rnn = tf.compat.v1.nn.rnn_cell.GRUCell(hiddenSz)

        if self.special_doi and self.pretrained:
            initialState = pre_state  # load pretrained state
        else:
            initialState = rnn.zero_state(batchSz, tf.float32)

        ff_inpt, nextState = tf.compat.v1.nn.dynamic_rnn(rnn, embed, initial_state=initialState)

        # update sum of the weights for l2regularization
        if self.l2regularization:
            sum_weights = sum_weights + tf.nn.l2_loss(nextState)

        if self.LDA_topics:  # if including LDA topics as input to the
            # feedforward layers
            ff_inpt = tf.concat([lda_inpt, output], 0)  # concatenate topic
            # contribution estimates to the output of the GRU cell

        if self.authorship:
            if not self.special_doi or not self.pretrained:
                author_embed = tf.Variable(tf.random.normal([len(author_list), author_embedSz], stddev=0.1))
                # TODO: add the author listing (author_list) to the NN
                # pre-processing function

            ff_inpt = tf.concat([ff_inpt, author_embed])

        l1Sz = hiddenSz
        if self.LDA_topics:
            lda_inpt_size = tf.size(input=lda_inpt)
            l1Sz += lda_inpt_size

        if self.authorship:
            l1Sz += author_embedSz

        if not self.special_doi:  # sentiment analysis (pos/neut/neg)
            # create weights and biases for three feedforward layers
            W1 = tf.Variable(tf.random.normal([l1Sz, ff1Sz], stddev=0.1))
            b1 = tf.Variable(tf.random.normal([ff1Sz], stddev=0.1))
            l1logits = tf.nn.relu(tf.tensordot(ff_inpt, W1, [[2], [0]]) + b1)
            l1Output = tf.nn.dropout(l1logits, 1 - (DOutRate))  # apply dropout
            W2 = tf.Variable(tf.random.normal([ff1Sz, ff2Sz], stddev=0.1))
            b2 = tf.Variable(tf.random.normal([ff2Sz], stddev=0.1))
            l2Output = tf.nn.relu(tf.tensordot(l1Output, W2, [[2], [0]]) + b2)
            W3 = tf.Variable(tf.random.normal([ff2Sz, 3], stddev=0.1))
            b3 = tf.Variable(tf.random.normal([3], stddev=0.1))
            # NOTE: Remember to adjust dimensions for the last layer if trinary
            # classification is not the goal of the neural network

            # update parameter vector lengths for l2regularization
            if self.l2regularization:
                for vector in [W1, b1, W2, b2, W3, b3]:
                    sum_weights = sum_weights + tf.nn.l2_loss(vector)

            ## calculate loss

            # calculate logits
            logits = tf.tensordot(l2Output, W3, [[2], [0]]) + b3

            # calculate sequence cross-entropy loss
            xEnt = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=answr)

            if self.l2regularization:
                loss = tf.reduce_mean(input_tensor=xEnt) + (alpha * sum_weights)
            else:
                loss = tf.reduce_mean(input_tensor=xEnt)

        elif self.special_doi:  # classification for a DOI
            if not self.pretrained:  # if initializing parameters
                # create weights and biases for three feedforward layers
                W1 = tf.Variable(tf.random.normal([l1Sz, ff1Sz], stddev=0.1))
                b1 = tf.Variable(tf.random.normal([ff1Sz], stddev=0.1))
                l1logits = tf.nn.relu(tf.matmul(ff_inpt, W1) + b1)
                l1Output = tf.nn.dropout(l1logits, 1 - (keepP))  # apply dropout
                W2 = tf.Variable(tf.random.normal([ff1Sz, ff2Sz], stddev=0.1))
                b2 = tf.Variable(tf.random.normal([ff2Sz], stddev=0.1))
                l2Output = tf.nn.relu(tf.matmul(l1Output, W2) + b2)
                W3 = tf.Variable(tf.random.normal([ff2Sz, 3], stddev=0.1))
                b3 = tf.Variable(tf.random.normal([3], stddev=0.1))

            elif self.pretrained:  # if using pre-trained weights
                W1 = tf.Variable(weights1)
                b1 = tf.Variable(biases1)
                l1logits = tf.nn.relu(tf.matmul(nextState, W1) + b1)
                l1Output = tf.nn.dropout(l1logits, 1 - (keepP))  # apply dropout
                W2 = tf.Variable(weights2)
                b2 = tf.Variable(biases2)
                l2Output = tf.nn.relu(tf.matmul(l1Output, W2) + b2)
                W3 = tf.Variable(weights3)
                b3 = tf.Variable(biases3)
                # NOTE: Remember to adjust dimensions for the last layer if trinary
                # classification is not the goal of the neural network

            l3Output = tf.nn.relu(tf.matmul(l2Output, W3) + b3)

            ### calculate loss

            # calculate logits
            logits = tf.matmul(l2Output, W3) + b3

            # softmax
            prbs = tf.nn.softmax(logits)

            # calculate cross-entropy loss
            xEnt = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(answr))

            if self.l2regularization:
                loss = tf.reduce_mean(input_tensor=(xEnt) + (NN_alpha * sum_weights))
            else:
                loss = tf.reduce_mean(input_tensor=xEnt)

            # calculate accuracy
            numCorrect = tf.equal(tf.argmax(input=prbs, axis=1), tf.argmax(input=answr, axis=1))
            numCorrect = tf.reduce_sum(input_tensor=tf.cast(numCorrect, tf.float32))

        ## training with AdamOptimizer

        train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        ## create the session and initialize the variables

        config = tf.compat.v1.ConfigProto(device_count=self.device_count)
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.compat.v1.global_variables_initializer())
        if not pretrained:
            state = sess.run(initialState)

    def Get_Set_Lengths(self):
        for set_key in self.set_key_list:
            for i, x in enumerate(self.indices[set_key]):
                self.lengths[set_key].append(len(self.indices[set_key][i]))
            Max[set_key] = max(lengths[set_key])  # max length of a post in a set
        self.Max_l = max(Max['train'], Max['dev'], Max['test'])
        # Max_l: max length of a comment in the whole dataset

    # TODO: add LDA input
    def train_and_evaluate(self):

        print("Number of planned training epochs: " + str(epochs))

        for k in range(epochs):  # for each epoch

            # timer
            print("Started epoch " + str(k + 1) + " at " + time.strftime('%l:%M%p, %m/%d/%Y'))

            for set_key in self.set_key_list:  # for each set

                TotalCorr = 0  # reset number of correctly classified examples

                # initialize vectors for feeding data and desired output
                inputs = np.zeros([self.batchSz, Max_l])
                if self.LDA_topics:
                    lda_inpt = np.zeros([self.num_topics, 1])
                if self.authorship:
                    # TODO: Calculate the number of assumed authors based on data
                    authorship_inpt = np.zeros([None, 1])
                answers = np.zeros([batchSz, 3], dtype=np.int32)

                # batch counters
                j = 0  # batch comment counter
                p = 0  # batch counter

                for i in range(len(self.indices[set_key])):  # for each comment in the set
                    inputs[j, :self.lengths[set_key][i]] = self.indices[set_key][i]
                    # TODO: fix this to pick up the human ratings from Qualtrics
                    if special_doi == True:
                        answers[j, :] = self.vote[set_key][i]
                    else:
                        answers[j, :] = self.sentiments[set_key][i]

                    j += 1  # update batch comment counter
                    if j == batchSz - 1:  # if the current batch is filled

                        if set_key == 'train':
                            # train on the examples
                            _, outputs, next, _, Corr = sess.run([train, output, nextState, loss, numCorrect],
                                                                 feed_dict={inpt: inputs, answr: answers,
                                                                            DOutRate: self.keepP})
                        else:
                            # test on development or test set
                            _, Corr = sess.run([loss, numCorrect],
                                               feed_dict={inpt: inputs, answr: answers, DOutRate: 1})

                        j = 0  # reset batch comment counter
                        p += 1  # update batch counter

                        # reset the input/label containers
                        inputs = np.zeros([self.batchSz, self.Max_l])
                        if self.LDA_topics:
                            lda_inpt = np.zeros([self.num_topics, 1], dtype=np.float32)
                        if self.authorship:
                            authorship_inpt = np.zeros([self.num_topics])
                            # TODO: fix the size of the vector and add indexing of the authors to the NN preprocessing
                        answers = np.zeros([self.batchSz, 3], dtype=np.int32)

                        # update the GRU state
                        state = next  # update the GRU state

                        # update total number of correctly classified examples or total loss based on the processed batch
                        TotalCorr += Corr

                    # Every 10000 comments or at the end of training, save the
                    # weights
                    if set_key == 'train' and ((i + 1) % 10000 == 0 or
                                               i == len(self.indices['train']) - 1):

                        # retrieve learned weights
                        if self.authorship:
                            word_embed, author_embed, weights1, weights2, weights3, biases1, biases2, biases3 = sess.run(
                                [E, author_embed, W1, W2, W3, b1, b2, b3])
                        else:
                            word_embed, weights1, weights2, weights3, biases1, biases2, biases3 = sess.run(
                                [E, W1, W2, W3, b1, b2, b3])

                        word_embed = np.asarray(word_embed)
                        if self.authorship:
                            author_embed = np.asarray(author_embed)
                        outputs = np.asarray(outputs)
                        weights1 = np.asarray(weights1)
                        weights2 = np.asarray(weights2)
                        weights3 = np.asarray(weights3)
                        biases1 = np.asarray(biases1)
                        biases2 = np.asarray(biases2)
                        biases3 = np.asarray(biases3)
                        # define a list of the retrieved variables
                        if self.authorship:
                            weights = ["word embeddings", "author embeddings", "state", "weights1", "weights2",
                                       "weights3", "biases1", "biases2", "biases3"]
                        else:
                            weights = ["word embeddings", "state", "weights1", "weights2", "weights3", "biases1",
                                       "biases2", "biases3"]
                        # write them to file
                        for variable in weights:
                            np.savetxt(output_path + "/" + variable, eval(variable))

                    # calculate set accuracy for the current epoch and save the value
                    self.accuracy[set_key][k] = float(TotalCorr) / float(p * self.batchSz)
                    print("Accuracy on the " + set_key + " set (Epoch " + str(k + 1) + "): " + str(
                        self.accuracy[set_key][k]))
                    print("Accuracy on the " + set_key + " set (Epoch " + str(k + 1) + "): " + str(
                        self.accuracy[set_key][k]), file=perf)

            ## early stopping
            if self.early_stopping:
                # if development set accuracy is decreasing, stop training to prevent overfitting
                if k != 0 and accuracy['dev'][k] < accuracy['dev'][k - 1]:
                    break

        # timer
        print("Finishing time:" + time.strftime('%l:%M%p, %m/%d/%Y'))
