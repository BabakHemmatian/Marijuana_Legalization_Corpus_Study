# import required packages
from numpy import random
from pathlib2 import Path
import csv

## Set paramters

year = '2018' # the year from which random posts will be sampled to evaluate the regex
sample_size = 100 # number of posts that will be sampled from each year to evaluate the regex
trial = 7 # the regex trial id

# determine the files with the combined text of comments and their indices
foriginal = "original_regex_" + str(year) + "_" + str(trial)
main_indices = "indices_regex_" + str(year) + "_" + str(trial)

# Combine relevant posts from [year] according to regex, or load existing compiled data
if Path(foriginal).is_file() and Path(main_indices).is_file():
    with open(foriginal,"r") as original:
        num_rel_comm = 0
        for line in original:
            num_rel_comm += 1
else:
    with open(foriginal,"w") as original, open(main_indices,"w") as main_indices:
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            with open('orig_'+ year + '_' + month + '_' + str(trial),"r") as original_month, open('idx_' + year + '_' + month + '_' + str(trial),"r") as indices_month:
                for line in original_month:
                    print(line.strip(),file=original)
                for line in indices_month:
                    print(line.strip(),file=main_indices)

# calculate the number of relevant posts for sampling
with open(foriginal) as original:
    num_rel_comm = 0
    for line in original:
        num_rel_comm += 1

## Sample [sample_size] comments for evaluating the Regex

sampled = random.choice(range(num_rel_comm), size=sample_size, replace=False)

## write a sample of regexed comments to a CSV file for evaluation

with open("sampled_" + str(year) + '_' + str(trial) + '.csv','a+') as sample, open(foriginal,"r") as original: # create the file

    writer_R = csv.writer(sample) # initialize the CSV writer
    writer_R.writerow(['number','relevance','text']) # write headers to the CSV file

    counter = 0
    # iterate over the regexed posts and write the sampled ones to file
    for idx,line in enumerate(original):
        if idx in sampled:
            counter += 1
            writer_R.writerow([counter,"",line]) # the indexing should be in line with the other output files
