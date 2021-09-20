# import required packages
import json
import bz2
import re
import os
import sys
import html
import time

## Set paramters

# NOTE: For best performance, do not run more iterations than the number of
# physical cores at the same time

year = '2018' # the year from which random posts will be sampled to evaluate the regex
months = ['11','12'] # Strings. 1-digit numbers should be preceded by a 0
trial = 7 # the regex trial id

## Get the regex

# get the list of words relevant to legality and marijuana from disk
# (requires marijuana.txt and legality.txt to be located in the same directory)
legality = []
marijuana = []
with open("legality_" + str(trial) + ".txt",'r') as f:
    for line in f:
        legality.append(re.compile(line.lower().strip()))

with open("marijuana_" + str(trial) + ".txt",'r') as f:
    for line in f:
        marijuana.append(re.compile(line.lower().strip()))

decoder = json.JSONDecoder() # define the json decoder object

main_counter=0 # id across all posts from that year
num_rel_comm=0 # id across relevant posts from that year

## prepare files

print("Started parsing at " + time.strftime('%l:%M%p')) # timer

for month in months:

    filename= 'RC_'+str(year)+'-'+month+'.bz2' # get the relevant compressed data file name

    with bz2.BZ2File(filename,'rb') as fin, open('orig_'+ year + '_' + month + '_' + str(trial),"w") as original, open('idx_' + year + '_' + month + '_' + str(trial),"w") as indices: # decompress the relevant bz2 file

        print("Started parsing month " + str(month) + " at " + time.strftime('%l:%M%p')) # timer

        # read data
        for line in fin: # for each comment
            main_counter += 1 # update the general counter

            # parse the json, and turn it into regular text
            comment = line.decode("utf-8")
            comment = decoder.decode(comment)
            original_body = html.unescape(comment["body"]) # original text
            # TODO: The normalized version should include removal of non-alphanumeric things. *ban* is ban, and we want the matching to reflect that to be comprehensive enough
            # TODO: Instead of normalization I should just add the plural, etc. cases to my \\b words

            # filter comments by relevance to the topic using the provided regex
            if any(not exp.search(original_body) is None for exp in marijuana) and any(not exp.search(original_body) is None for exp in legality):

                # update the relevance counter
                num_rel_comm += 1

                # Write the counters and the comment to disk
                original_body = original_body.replace("\n","") # remove mid-comment lines
                print(" ".join(original_body.split()),file=original) # record the original comment
                print(main_counter,file=indices) # record the main index

# TODO: Normalized version

print("Finished parsing at " + time.strftime('%l:%M%p')) # timer

# print out the number of relevant comments and the total number of comments
print("number of relevant comments: " + str(num_rel_comm))
print("total number of comments from " + str(year) + ": " + str(main_counter))
