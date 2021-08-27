# import necessary modules
from config import *
import time
import numpy
from pathlib2 import Path

# define set of subreddits to sample from
# BUG: currently only works for individual subreddits.
subreddits = ['changemyview','unpopularopinion']
# determine number of documents to sample per subreddit
num_sample = 10
min_length = 500 # number of tokens

start = time.time()  # measure processing time

# create dictionaries for storing the indices of relevant posts, as well as their texts
indices = {}
comments = {}
lengths = {}
for subreddit in subreddits:
    indices[subreddit] = []
    comments[subreddit] = []
    lengths[subreddit] = []

# checks for missing files for the specificied subreddits
missing_files = []
for subreddit in subreddits:
    if not Path(model_path+"/subreddit/subreddit-{}".format(subreddit)).is_file():
        missing_files.append(subreddit)

# if some subreddits haven't been previously extracted, extract indices
if len(missing_files) != 0:

    # extracts indices of relevant posts and stores them in memory
    with open(model_path+"/subreddit/subreddit","r") as subreddit_file:
        for idx,line in enumerate(subreddit_file):
            if line.strip() in subreddits:
                indices[line.strip()].append(idx)
    print("Number of relevant documents found: ")
    for subreddit in subreddits:
        print(subreddit+": "+str(len(indices[subreddit])))

    # uses the list of indices to extract comment texts from the full dataset
    with open(model_path+"/original_comm/original_comm", "r") as original_comm:
        for subreddit in subreddits:
            for idx,line in enumerate(original_comm):
                if idx in indices[subreddit]:
                    comments[subreddit].append(line.strip())
                    lengths[subreddit].append(len(line.strip().split())-1)

    # writes the indices and texts to subreddit-specific files
    for subreddit in subreddits:
        with open(model_path+"/subreddit/subreddit-{}".format(subreddit),"w") as extracted:
            for idx,element in enumerate(indices[subreddit]):
                print(str(element)+";;; "+comments[subreddit][idx],file=extracted)

    # Calculates and reports processing time
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Subreddit-specific posts extracted and written to files in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes),
                                                                           seconds))
else: # if files exist for all listed subreddits, read their content
    print("Files from previous runs found. Loading.")
    for subreddit in subreddits:
        with open(model_path+"/subreddit/subreddit-{}".format(subreddit),"r") as extracted:
            for line in extracted:
                indices[subreddit].append(int(line.split(";;;")[0].strip()))
                comments[subreddit].append(line.split(";;;")[1].strip())
                lengths[subreddit].append(len(line.strip().split())-1)

# for each subreddit, extract [num_sample] comments as a random sample and print
# them along with their index in the main dataset

long_indices = {}
for subreddit in subreddits:
    long_indices[subreddit] = []
    for idx,post in enumerate(indices[subreddit]):
        if lengths[subreddit][idx] >= min_length:
            long_indices[subreddit].append(idx)

rand_subsample = []
for subreddit in subreddits:
    print("Extracting random sample of {} posts from the {} subreddit".format(num_sample,subreddit))
    with open(model_path+"/subreddit/subreddit-"+subreddit,"r") as extracted:
        random_sample = numpy.random.choice(long_indices[subreddit],size=num_sample*len(subreddits),replace=False)
        for idx,line in enumerate(extracted):
            if idx in random_sample:
                rand_subsample.append(line)

numpy.random.shuffle(rand_subsample)
with open(model_path+"/subreddit/subreddits_rand_"+str(num_sample), "w") as sample_file:
    for post in rand_subsample:
        print(post,file=sample_file)
