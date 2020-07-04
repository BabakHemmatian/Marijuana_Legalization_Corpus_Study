# import needed modules
import bz2
from defaults import *
from reddit_parser import Parser
import lzma
import zstandard as zstd
import json
import os
from collections import Counter
import html
import time
from pathlib2 import Path
import argparse

# grab the machine argument from slurm if running on the cluster.
# NOTE: If running locally, comment out the following chuck and set machine to
# 'local'
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--machine', type = str)
    args = argparser.parse_args()
machine = args.machine

# create directory for subreddits
if not os.path.exists(model_path+"/subreddit/"):
    print("Creating directory to store the output")
    os.makedirs(model_path+"/subreddit/")

# override the dates determined in defaults.py as needed
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2008,2020) # year range
for year in years:
    for month in months:
        dates.append((year,month))

# Format dates to be consistent with pushshift file names
def format_date(yr, mo):
    if len(str(mo)) < 2:
        mo = '0{}'.format(mo)
    assert len(str(yr)) == 4
    assert len(str(mo)) == 2
    return "{}-{}".format(yr, mo)

## Raw Reddit data filename format. The compression types for dates handcoded
# based on https://files.pushshift.io/reddit/comments/

# IMPORTANT: Remember to recode the filenames below accordingly if the RC files
# have been downloaded and transformed into a different compression type than
# shown to prevent filename errors

def get_rc_filename(yr, mo):
    date = format_date(yr, mo)
    if (yr == 2017 and mo == 12) or (yr == 2018 and mo < 10):
        return 'RC_{}.xz'.format(date)
    elif (yr == 2018 and mo >= 10) or (yr > 2018):
        return 'RC_{}.zst'.format(date)
    else:
        return 'RC_{}.bz2'.format(date)

# define the json decoder object
decoder = json.JSONDecoder()

# go through the pushshift dataset for each month and grab the subreddits
# NOTE: Assumes that the files are on disk within data_path
for yr,mo in dates: # for each month

    filename = get_rc_filename(yr,mo) # get the correct data file name

    print("Initiating processing of " + filename + " at "
          + time.strftime('%l:%M%p, %m/%d/%Y')) # timer

    # check for existing results
    if not Path(model_path+"/subreddit/subreddit-{}-{}".format(yr,mo)).is_file():

        # if none exists, grab the indices of relevant posts
        mo_main_indices = []
        with open(model_path+"/original_indices/original_indices-{}-{}".format(yr,mo),"r") as indices:
            for line in indices:
                if line.strip() != "":
                    mo_main_indices.append(int(line.strip()))

        # open the file as a text file, in utf8 encoding, based on encoding type
        if '.zst' in filename:
            file = open(data_path + filename, 'rb')
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(file)
            fin = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='ignore')
        elif '.xz' in filename:
            fin = lzma.open(data_path + filename, 'r')
        elif '.bz2' in filename:
            fin = bz2.BZ2File(data_path + filename, 'r')
        else:
            raise Exception('File format not recognized')

        # read data
        per_file_container = [] # container for running on the cluster
        for idx,line in enumerate(fin):  # for each comment
            idx += 1 # correct the index

            if idx in mo_main_indices: # if comment is identified as relevant

                if '.zst' not in filename:
                    line = line.decode('utf-8','ignore')

                # decode and grab subreddit information
                comment = decoder.decode(line)
                subreddit = html.unescape(comment["subreddit"]).strip()

                # if running locally, write to disk immediately to preserve memory
                if machine == "local":
                    with open(model_path + "/subreddit/subreddit-{}-{}".format(yr,mo),"a+") as monthly_file:
                        monthly_file.write(subreddit + "\n")
                    with open(model_path + "/subreddit/subreddit".format(yr,mo),"a+") as general_file:
                        general_file.write(subreddit + "\n")
                else: # if running on the cluster, add to a list to dump to disk
                # at the end of the month
                    per_file_container.append(subreddit)

        # dump to disk at the end of the month if running on the cluster
        if not machine == "local":
            # monthly file
            with open(model_path + "/subreddit/subreddit-{}-{}".format(yr,mo),"a+") as monthly_file:
                for element in per_file_container:
                    if element.strip() != "":
                        monthly_file.write(element.strip() + "\n")

            # aggregate file
            with open(model_path + "/subreddit/subreddit","a+") as general_file:
                for element in per_file_container:
                    if element.strip() != "":
                        general_file.write(element.strip() + "\n")

    print("Finished processing of " + filename + " at "
          + time.strftime('%l:%M%p, %m/%d/%Y')) # timer

# Gather subreddit counts
subreddit_counts = {}
counter = 0
with open(model_path + "/subreddit/subreddit","r") as general_file:
    for line in general_file:
        if line.strip() != "":
            counter += 1
        if line.strip() in subreddit_counts.keys():
            subreddit_counts[line.strip()] += 1
        else:
            subreddit_counts[line.strip()] = 1

# print counts
k = Counter(subreddit_counts)
# Finding 5 highest values
high = k.most_common(5)
print(counter)
print(high)
