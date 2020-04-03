from pathlib2 import Path
from subprocess import check_output
import sys
import time
from config import *
import reddit_parser

### Function for writing parameters and model performance to file
## TODO: Write a separate set of variables to file for NN
def Write_Performance(output_path=output_path, NN=NN):
    with open(output_path+"/Performance",'a+') as perf:
        if not NN:
            print("***",file=perf)
            print("Time: "+time.strftime("%Y-%m-%d %H:%M:%S"),file=perf)
            print("*** Hyperparameters ***", file=perf)
            print("Training fraction = " + str(training_fraction),file=perf)
            print("Maximum vocabulary size = " + str(MaxVocab),file=perf)
            print("Minimum number of documents a token can appear in and be included = " + str(no_below),file=perf)
            print("Fraction of documents, tokens appearing more often than which will be filtered out = " + str(no_above),file=perf)
            print("Number of topics = " + str(num_topics),file=perf)
            print("Fraction of topics sampled = " + str(sample_topics),file=perf)
            print("Number of top words recorded for each topic = " + str(topn),file=perf)
            print("Number of comments sampled from each top topic = " + str(sample_comments),file=perf)
            print("Minimum comment length for sampled comments = " + str(min_comm_length),file=perf)
            print("Alpha (LDA) = " + str(alpha),file=perf)
            print("Eta (LDA) = " + str(eta),file=perf)
            print("One-hot topic contribution calculation = " + str(one_hot_topic_contributions),file=perf)
            print("Topic idf inclusion in contribution calculation = " + str(topic_idf),file=perf)
            print("Topic idf frequency counter threshold = " + str(topic_idf_thresh),file=perf)
            print("Minimum topic probability = " + str(minimum_probability),file=perf)
            print("Minimum term probability = " + str(minimum_phi_value),file=perf)

        else: # if running a neural network analysis

            # record the pre-processing paramters
            print("Training fraction = " + str(NN_training_fraction),file=perf)
            print("Vocabulary size = " + str(MaxVocab),file=perf)
            print("Frequency filter = below " + str(FrequencyFilter),file=perf)

            # record the kind of network
            print("***",file=perf)
            print("special_doi = " + str(special_doi),file=perf)
            print("pretrained = " + str(pretrained),file=perf)

            # record the hyperparameters
            print("Number of epochs: "+str(epochs),file=perf)
            print("Learning_rate = " + str(learning_rate),file=perf)
            print("Batch size = " + str(batchSz),file=perf)
            ##TODO: should this be author_embedSz or word_embedSz?
            print("Embedding size = " + str(word_embedSz),file=perf)
            print("Recurrent layer size = " + str(hiddenSz),file=perf)
            print("1st feedforward layer size = " + str(ff1Sz),file=perf)
            print("2nd feedforward layer size = " + str(ff2Sz),file=perf)
            print("Dropout rate = " + str(1 - keepP),file=perf)
            print("L2 regularization = " + str(l2regularization),file=perf)
            print("L2 regularization constant = " + str(alpha),file=perf)
            print("Early stopping = " + str(early_stopping),file=perf)

### calculate the yearly relevant comment counts
def Get_Counts(path=path, random=False, frequency="monthly"):
    assert frequency in ("monthly", "yearly")

    fns=reddit_parser.Parser().get_parser_fns()
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

def essentially_eq(a, b):
    return abs(a-b)<= 0.1
