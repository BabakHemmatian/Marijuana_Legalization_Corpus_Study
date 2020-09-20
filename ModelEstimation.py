from collections import defaultdict, OrderedDict
import csv
from functools import partial
import ast
import operator
import numpy as np
import tensorflow as tf
import gensim
import glob
import pickle
from math import ceil, floor
import matplotlib.pyplot as plt
import multiprocessing
import os
from pathlib2 import Path
import random
import time
from config import *
from reddit_parser import Parser
import tensorflow as tf
import sqlite3
parser_fns = Parser().get_parser_fns()
from simpletransformers.classification import ClassificationModel
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaModel, pipeline
from Utils import *
# IDEA: # We should check this for LM pretraining on our dataset,
# and maybe do that for the ROBERTA section later:
# https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py

## converter functions for storing and retrieving numpy arrays from the SQLite database
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


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

# TODO: This object should be updated based on what's shared bw the models
class ModelEstimator(object):
    def __init__(self, all_=ENTIRE_CORPUS, MaxVocab=MaxVocab,
                 output_path=output_path, path=model_path, dates=dates,
                 DOI=DOI, training_fraction=training_fraction,
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
        self.DOI = DOI
        self.training_fraction = training_fraction
        self.V = V  # vocabulary #TODO: check to see if this is needed

    # TODO: first set aside 10 percent of data for test, then determine the
    # validation split
    ### function to determine comment indices for new training, development and test sets
    def Create_New_Sets(self, indices, human_ratings_pattern):
        print("Creating sets")

        # determine number of comments in the dataset
        if self.all_:

            if NN and self.DOI is not None:

                # check to see if human comment ratings can be found on disk
                files = []
                info_files = []
                for element in human_ratings_pattern:
                    files_to_add = glob.glob(self.path + element)
                    info_to_add = glob.glob(re.sub(r'ratings', 'info', self.path + element))
                    for file in files_to_add:
                        files.append(file)
                    for file in info_to_add:
                        info_files.append(file)

                if len(files) == 0:
                    raise Exception("Human comment ratings for DOI training could not be found on disk.")
                if len(info_files) == 0:
                    raise Exception("Metadata files for human comment ratings could not be found on disk.")

                human_ratings = {"attitude":{},"persuasion":{}}
                for file in files:

                    # retrieve the number of comments for which there are complete human ratings
                    with open(file, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # read human data for sampled comments one by one
                        for idx, row in enumerate(reader):
                            # ignore headers and record the index of comments that are interpretable and that have ratings for all three goal variables
                            if (idx != 0) and (row[2] != '0'): # if not a header or an irrelevant comment

                                relevant_rows = [row[3],row[4]]

                                for id_,relevant_row in enumerate(relevant_rows):
                                    # print("rows")
                                    # print(row[3], row[4])

                                    formatted_row = relevant_row
                                    if "//" in formatted_row:
                                        formatted_row = relevant_row.split("//")[0]

                                    if "unclear" in formatted_row.lower():
                                        if any(char.isdigit() for char in formatted_row):
                                            formatted_row = int(re.sub('[^0-9]','', formatted_row))
                                        else:
                                            formatted_row = 0

                                    if id_ == 0:
                                        if row[0] not in human_ratings["attitude"]:
                                            if relevant_row.strip() != "":
                                                human_ratings["attitude"][int(row[0])] = [formatted_row]
                                        else:
                                            human_ratings["attitude"][int(row[0])].append(formatted_row)
                                    else:
                                        if row[0] not in human_ratings["persuasion"]:
                                            if relevant_row.strip() != "":
                                                human_ratings["persuasion"][int(row[0])] = [formatted_row]
                                        else:
                                            human_ratings["persuasion"][int(row[0])].append(formatted_row)

                assert len(human_ratings["attitude"]) == len(human_ratings["persuasion"])

                info_indices = {}
                for id_ in human_ratings["attitude"].keys():
                    if id_ not in info_indices.keys():
                        for file in info_files:
                            with open(file,"r") as csvfile:
                                reader = csv.reader(csvfile)
                                count = 0
                                for row in reader:
                                    if count!= 0:
                                        if int(row[0].strip()) == id_:
                                            if not row[5].isdigit():
                                                info_indices[int(row[0].strip())] = [int(i) for i in row[5].strip().split(",")]
                                            else:
                                                info_indices[int(row[0].strip())] = int(row[5].strip())
                                    count = count + 1

                assert len(info_indices) == len(human_ratings["attitude"])

                num_comm = len(human_ratings["attitude"])  # the number of valid samples for network training
                indices = human_ratings["attitude"].keys()  # define sets over sampled comments with human ratings

            elif self.NN:
                num_comm = list(indices)[-1]  # retrieve the total number of comments
                indices = range(num_comm)  # define sets over all comments

        else:  # if using LDA on a random subsample of the comments
            num_comm = len(indices)  # total number of sampled comments

        num_train = int(ceil(0.90 * num_comm))  # size of training set

        training_set = []
        testing_set = []

        if isinstance(self, NNModel):  # for NN
            num_test = num_comm - num_train  # the number of comments in development set or test set

            self.sets['test'] = random.sample(indices, num_test)  # choose development comments at random
            self.sets['train'] = set(indices).difference(self.sets['test'])

            # sort the indices based on position in the database
            for set_key in self.set_key_list:
                self.sets[set_key] = sorted(list(self.sets[set_key]))

            # Check test set came out with the right proportion
            assert len(self.sets['test']) + len(self.sets['train']) == len(
                indices), "The sizes of the training, development and test sets do not add up to the number of posts on file"

            # write the sets to file
            for set_key in self.set_key_list:
                np.save(self.path + '/' + set_key + '_set_' + str(self.DOI), self.sets[set_key])

            training_set = self.sets['train']
            testing_set = self.sets['test']

        else:  # for LDA over the entire corpus
            num_eval = num_comm - num_train  # size of evaluation set

            self.LDA_sets['eval'] = random.sample(indices, num_eval)  # choose evaluation comments at random
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
                        print(index, file=f)

        # TODO: here's where we should add both the training and test set membership to
        # the SQL database, and upload the actual ratings to the attitude or persuasion

        #In the database we should add training/test column (ALTER TABLE table_name ADD training int)

        # rows_to_be_added = {} #key is the index and value is a list with attitude, persuasion, training/test value
        # attitude_key_set = human_ratings["attitude"].keys()
        # persuasion_key_set = human_ratings["persuasion"].keys()
        # training_key_set = training_set
        # test_key_set = testing_set

        # for aks in attitude_key_set:
        #     original_index = -1
        #     if type(info_indices[aks]) is list:
        #         original_index = info_indices[aks][0]
        #     else:
        #         original_index = info_indices[aks] + 1
        #     set_values = set(human_ratings["attitude"][aks])
        #     val = ",".join(str(s) for s in set_values)
        #     rows_to_be_added[original_index] = [val]

        # for pks in persuasion_key_set:
        #     original_index = -1
        #     if type(info_indices[pks]) is list:
        #         original_index = info_indices[pks][0]
        #     else:
        #         original_index = info_indices[pks] + 1
        #     set_values = set(human_ratings["persuasion"][pks])
        #     val = ",".join(str(s) for s in set_values)
        #     if original_index in rows_to_be_added.keys():
        #         rows_to_be_added[original_index].append(val)
        #     else:
        #         rows_to_be_added[original_index] = ["", val]

        # for trks in training_key_set:
        #     original_index = -1
        #     if type(info_indices[trks]) is list:
        #         original_index = info_indices[trks][0]
        #     else:
        #         original_index = info_indices[trks] + 1
        #     val = 1
        #     if original_index in rows_to_be_added.keys():
        #         rows_to_be_added[original_index].append(val)
        #     else:
        #         rows_to_be_added[original_index] = ["", "", val]

        # for teks in test_key_set:
        #     original_index = -1
        #     if type(info_indices[teks]) is list:
        #         original_index = info_indices[teks][0]
        #     else:
        #         original_index = info_indices[teks] + 1
        #     val = 0
        #     if original_index in rows_to_be_added.keys():
        #         rows_to_be_added[original_index].append(val)
        #     else:
        #         rows_to_be_added[original_index] = ["", "", val]

        # conn = sqlite3.connect("reddit_{}.db".format(self.num_topics))
        # cursor = conn.cursor()
        # for row in rows_to_be_added.keys():
        #     print("row", row)
        #     sql = "UPDATE comments SET attitude={0}, persuasion={1}, training={2} WHERE original_indices={3}".format(rows_to_be_added[row][0], rows_to_be_added[row][1], rows_to_be_added[row][2], row)
        #     cursor.execute(sql)
        # conn.commit()

    # NOTE: The lack of an evaluation set for NN should reflect in this func too
    ### function for loading, calculating, or recalculating sets
    def Define_Sets(self,human_ratings_pattern=None):
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
                Path(self.fns["test_set"]).is_file()):

            # determine if the comments and their relevant indices should be deleted and re-initialized or the sets should just be loaded
            Q = input("Comment sets are already available. Do you wish to delete them and create new ones [Y/N]?")

            # If recreating the sets is requested, delete the current ones and reinitialize
            if Q == "Y" or Q == "y":
                print("Deleting any existing sets.")

                # delete previous record
                for set_key in self.set_key_list:
                    if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["indexed_{}_set".format(set_key)])
                    if Path(self.fns["{}_set".format(set_key)]).is_file():
                        os.remove(self.fns["{}_set".format(set_key)])

                self.Create_New_Sets(indices,human_ratings_pattern)  # create sets

            # If recreating is not requested, attempt to load the sets
            elif Q == "N" or Q == "n":
                # if the sets are found, load them
                if (Path(self.fns["train_set"]).is_file()
                        and Path(self.fns["test_set"]).is_file()
                ):

                    print("Loading sets from file")
                    for set_key in self.set_key_list:
                        self.sets[set_key] = np.load(self.fns["{}_set".format(set_key)])

                    # ensure set sizes are correct
                    assert len(self.sets['test']) + len(self.sets['train']) == len(
                        indices), "The sizes of the training, development and test sets do not add up to the number of posts on file"

                else:  # if the sets cannot be found, delete any current sets and create new sets
                    print("Failed to load previous sets. Reinitializing")

                    # delete partial record
                    for set_key in self.set_key_list:
                        if Path(self.fns["indexed_{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["indexed_{}_set".format(set_key)])
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    self.Create_New_Sets(indices,human_ratings_pattern)  # create sets

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
                Path(self.fns["test_set"]).is_file()
            ) or (not isinstance(self, NNModel) and
                  Path(self.fns["train_set"]).is_file() and
                  Path(self.fns["eval_set"]).is_file()):

                print("Loading sets from file")

                if isinstance(self, NNModel):  # for NN
                    for set_key in self.set_key_list:
                        self.sets[set_key] = np.load(self.fns["{}_set".format(set_key)])

                    # ensure set sizes are correct
                    l = list(indices[-1]) if self.all_ else len(list(indices))
                    assert len(self.sets['test']) + len(self.sets['train']) == l, "The sizes of the training, development and test sets do not add up to the number of posts on file"

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
                    #FOR BABAK: check if this is okay
                    self.Create_New_Sets(indices, human_ratings_pattern)

                else:  # for LDA
                    # delete any partial set
                    for set_key in self.LDA_set_keys:
                        if Path(self.fns["{}_set".format(set_key)]).is_file():
                            os.remove(self.fns["{}_set".format(set_key)])

                    # create new sets
                    self.Create_New_Sets(indices, human_ratings_pattern)


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

# TODO: Some of the current fns can simply be loaded from the SQL database
# TODO: In all cases, data should be read from that database
class NNModel(ModelEstimator):
    def __init__(self, DOI=DOI, RoBERTa_model=RoBERTa_model,pretrained=pretrained,
                 FrequencyFilter=FrequencyFilter, learning_rate=learning_rate,
                 batch_size=batch_size, ff2Sz=ff2Sz, LDA_topics=LDA_topics,
                 # keepP=keepP, l2regularization=l2regularization, NN_alpha=NN_alpha,
                 num_topics=num_topics, early_stopping=early_stopping,
                 authorship=authorship, top_authors=top_authors,
                 use_subreddits=use_subreddits, top_subs=top_subs,
                 epochs=epochs, **kwargs):
        ModelEstimator.__init__(self, **kwargs)
        # TODO: define the truncation variable. It should be the default because
        # of RoBERTa's pretraining. Don't need to make it customizable
        self.DOI = DOI
        self.RoBERTa_model = RoBERTa_model
        self.pretrained = pretrained
        self.FrequencyFilter = FrequencyFilter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ff2Sz = ff2Sz
        self.LDA_topics = LDA_topics
        self.num_topics = num_topics
        self.authorship = authorship
        self.top_authors = top_authors
        self.use_subreddits = use_subreddits
        self.top_subs = top_subs
        # self.keepP = keepP
        # self.l2regularization = l2regularization
        # if self.l2regularization:
        #     self.NN_alpha = NN_alpha
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.set_key_list = ['train', 'test']  # for NN
        self.sets = {key: [] for key in self.set_key_list}  # for NN
        self.indices = {key: [] for key in self.set_key_list}
        self.lengths = {key: [] for key in self.set_key_list}
        self.Max = {key: [] for key in self.set_key_list}
        self.Max_l = None
        self.sentiments = {key: [] for key in self.set_key_list}  # for NN

        # TODO: the evaluation measures should probably be updated based on
        # recent changes to reddit_parser.py
        self.accuracy = {key: [] for key in self.set_key_list}
        for set_key in self.set_key_list:
            self.accuracy[set_key] = np.empty(epochs)

        self.fns = self.get_fns()

    # TODO: remove references to nn_prep. Remove those parts of this that won't
    # be used
    def get_fns(self, **kwargs):
        fns = {"counts": parser_fns["counts"] if self.all_ else parser_fns["counts_random"],
               "dictionary": "{}/dict_{}".format(self.path, self.DOI),
               "train_set": "{}/train_{}".format(self.path, self.DOI),
               "test_set": "{}/test_{}".format(self.path, self.DOI),
               # TODO: do we want to get different train/test sets for things other than DOI?
               "indexed_train_set": "{}/indexed_train_{}".format(self.path, self.DOI),
               "indexed_test_set": "{}/indexed_test_{}".format(self.path, self.DOI),
               }


        for k, v in kwargs.items():
            fns[k] = v
        return fns

    # TODO: adding the labels should be a separate function

    ### load or create vocabulary and load or create indexed versions of comments in sets
    # NOTE: Only for NN. For LDA we use gensim's dictionary functions
    def RoBERTa_Set(self):

        # load the SQL database
        try:
            if not LDA_topics:
                conn = sqlite3.connect("reddit.db",detect_types=sqlite3.PARSE_DECLTYPES)
            else:
                conn = sqlite3.connect("reddit_{}.db".format(num_topics),detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = conn.cursor()
        except:
            raise Exception('Pre-processed SQL database could not be found')

        ## record word frequency in the entire dataset
        with open(self.fns["counts"],"r") as f:
            for line in f:
                if line.strip() != 0:
                    total_count = int(line)
     
        cursor.execute("SELECT COUNT(*) AS CNTREC FROM pragma_table_info('comments') WHERE name='roberta_activation'")
        column = cursor.fetchall()

        print("batch_size", self.batch_size)

        if column[0][0] == 0:

            print("RoBERTa activations for the database not found. Computing and adding activations.")
            cursor.execute("ALTER TABLE comments ADD roberta_activation array")
            conn.commit()

            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            roberta = TFRobertaModel.from_pretrained('roberta-base')

            for i in range(total_count):
                t0 = time.time()
                
                if i != 0 and (i+1 % self.batch_size == 0 or i+1 == total_count):
                    print("i", i)
                    print("total count", total_count)

                    cursor.execute("SELECT rowid,original_comm,roberta_activation FROM comments WHERE rowid >= {} AND rowid <= {}".format(i+1,i+self.batch_size+1))

                    train_texts = []
                    train_indices = []

                    for comment in cursor:  # for each comment
                        train_indices.append(int(comment[0].strip()))
                        comment[1] = line.decode('utf-8','ignore')
                        train_texts.append(comment[1].strip())
                    print("finished comment")

                    cursor.execute("SELECT original_comm FROM comments")
                    texts = [item[0] for item in cursor.fetchall()]

                    print("finished execute")

                    encoded_input = tokenizer(texts, return_tensors="tf",truncation=True,padding=True,max_length=512)
                    print("finished toensizer")
                    roberta_output = roberta(encoded_input)
                    print("finished roberta")
                    roberta_output = np.asarray(roberta_output[0]) # shape (batch_size, 3, hidden_size)
                    
                    print("finished encoding")
                    if i+1 == total_count:
                        roberta_output = roberta_output.reshape(self.batch_size,2304)
                    else:
                        roberta_output = roberta_output.reshape(len(train_texts),2304) 
                    
                    print("finished reshaping")
                    for id_,document in enumerate(roberta_output):
                        for element in cursor.execute("SELECT roberta_activation FROM comments WHERE rowid = {}".format(id_+1)):
                            cursor.execute("UPDATE comments SET roberta_activation = {}".format(roberta_output))
                    conn.commit()

                    # sql_query = "INSERT INTO comments (rowid, roberta_activation) VALUES "
                    # for id_,document in enumerate(roberta_output):
                    #     sql_query += "(" + (id_+1) + "," + roberta_output + "),"
                    # sql_query = sql_query[:-1]
                    # cursor.execute(sql_query)
                    # conn.commit()

                t1 = time.time()
                print("Time taken", t1-t0)
            print("Finished processing the dataset using RoBERTa-base at " + time.strftime('%l:%M%p, %m/%d/%Y'))
        else:
            print("Loading RoBERTa-base activations from the database.")


    # TODO: Should rewrite this as a utility for the training function
    ## function for getting average sentiment values for training and test sets
    def Get_Human_Ratings(self, path):
        if not Path(self.fns["sentiments"]).is_file():
            raise Exception("Sentiment data could not be found on file.")
        else:
            with open(self.fns["sentiments"], "r") as sentiments:
                for idx, sentiment in enumerate(sentiments):
                    if idx in self.sets["train"]:
                        self.sentiments["train"].append(sentiment)
                    elif idx in self.sets["test"]:
                        self.sentiments["test"].append(sentiment)
                    else:
                        raise Exception("The set indices are not correctly defined")


    ## Function for creating the neural network's computation graph, training
    # and evaluating
    def train_and_evaluate(self):

        if not device_count is None:
            self.device_count = device_count

        input1 = tf.keras.layers.Input(shape = (2304,),dtype=tf.float32)
        inpt2_sz = 0
        if self.authorship:
            input2_sz += self.top_authors + 1
        if self.LDA_topics:
            input2_sz += self.num_topics
        if self.use_subreddits:
            input2_sz += self.top_subs + 1
        if inpt2_sz != 0:
            input2 = tf.keras.layers.Input(shape = (input2_sz,))
            ff1 = tf.keras.layers.Dense(128,dtype=tf.float32)([input1,input2])
        else:
            ff1 = tf.keras.layers.Dense(128,dtype=tf.float32)(input1)
        out = tf.keras.layers.Dense(3,dtype=tf.float32)(ff1)
        if inpt2_sz != 0:
            model = tf.keras.Model([input1,input2], [out])
        else:
            model = tf.keras.Model([input1], [out])

        model.compile(optimizer = tf.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',metrics=["mae", "acc"]) # TODO: can I replace the metrics here with what I want?

        ## create placeholders for input, output, loss weights and dropout rate
        # DOutRate = tf.compat.v1.placeholder(tf.float32)

        ## set up the graph parameters

        # TODO: fix this based on how Nate is saving
        # Should use these:     layer. get_weights(): returns the weights of the layer as a list of Numpy arrays.
        #                       layer. set_weights(weights): sets the weights of the layer from a list of Numpy arrays.

        # for pre-trained classification network, load parameters from file
        if self.pretrained:
            # TODO: needs to check disk for previous data and if not, train
            # TODO: add the loading of RoBERTa weights
            # TODO: fix the reference to the objects being "loaded"
            print("Loading parameter estimates from file")
            ff1.set_weights(np.load("ff1"))
            out.set_weights(np.load("out"))
        else:
            print("Initializing parameter estimates")
            ff1_rand_weights = np.random.normal(0,0.1,size=ff1.shape)
            ff1.set_weights(ff1_rand_weights)
            out_rand_weights = np.random.normal(0,0.1,size=ff1.shape)
            out.set_weights(out_rand_weights)

        # calculate sum of the weights for l2regularization
        # if self.l2regularization:
        #     sum_weights = tf.nn.l2_loss(ff1)
        #     sum_weights+ = tf.nn.l2_loss(out)
        #     # TODO: add for fine-tuning roberta
        #   losses = losses + (alpha * sum_weights)

    # TODO: add LDA input, etc.
    # TODO: add F1 to metrics


        # NEW STUFF
        checkpoint_path = self.output_path + "/params/training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # TODO: the input should be an iterator using fetchall. In fact, replace get_indexed_comment and index_set with some light processing here using iterators
        # TODO: this should be updated to reflect the configurable parts of the input, probably through an IF condition
        # TODO: should add rowid condition based on whether something is in the training set.

        conn = sqlite3.connect("reddit_{}.db".format(self.num_topics))
        cursor = conn.cursor()

        document_batch = []

        if input2_sz == 0:
            command = cursor.execute("SELECT roberta_activation,{} FROM comments WHERE {} IS NOT NULL".format(self.DOI,self.DOI))
        else:
            input2 = []
            if self.LDA_topics:
                for topic in range(num_topics):
                    input2.append("topic_{}".format(topic))
                if self.authorship:
                    input2.append("author")
                if self.subreddits:
                    input2.append("subreddit")

            command = cursor.execute("SELECT roberta_activation,{} FROM comments WHERE attitude IS NOT NULL".format(",".join(input2)))

        for document in command:
            document_batch.append(document)
            # IF any additions are needed as input2, get them from the database in the SQL command below
            # ELSE:

            if len(document_batch) == 10000:

                # TODO: the reformatting of the subreddits should come here
                # TODO: the null topic values should be fed in as zero
                model.fit(x = np.asarray(roberta_output[0]), y = np.asarray(roberta_output[1]), batch_size = batch_size, epochs = epochs, validation_split = 0.2, validation_batch_size=batch_size, metrics=[tf.keras.metrics.CategoricalAccuracy,tf.keras.metrics.Precision,tf.keras.metrics.Recall])

        # timer
        print("Finishing time:" + time.strftime('%l:%M%p, %m/%d/%Y'))

# TODO: add a testing function
