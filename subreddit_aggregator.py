# import needed modules
from defaults import *
import os
from collections import Counter
import argparse

# grab the machine argument from slurm if running on the cluster.
# NOTE: If running locally, comment out the following chuck and set machine to
# 'local'
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--machine', type = str)
    args = argparser.parse_args()
machine = args.machine

# aggregate file
dates=[] # initialize a list to contain the year, month tuples
months=range(1,13) # month range
years=range(2008,2020) # year range
for year in years:
    for month in months:
        dates.append((year,month))

for yr,mo in dates:
    with open(model_path + "/subreddit/subreddit-{}-{}".format(yr,mo),"r") as monthly_file, open(model_path + "/subreddit/subreddit","a+") as general_file:
        for element in monthly_file:
            if element.strip() != "":
                general_file.write(element.strip() + "\n")

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
