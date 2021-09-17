import csv
import sys
import numpy
from numpy import random

# create a dictionary for counting the non-junk topics
# (needs the cleaned sample of top comments, with 
# junk topics and identified junk posts removed)
topic_dict = {}
with open("cleaned_sample_keys-f.csv",'r+') as csvfile:
    reader = csv.reader(csvfile)
    # read human data for sampled comments one by one
    for idx,row in enumerate(reader):
        if idx != 0:
            if row[4] not in topic_dict.keys():
                topic_dict[row[4]] = 0

# sample 12 highly-representative posts for each included topic
resampled = []
with open("cleaned_sample_keys-f.csv",'r+') as csvfile:
    reader = csv.reader(csvfile)
    # read human data for sampled comments one by one
    for idx,row in enumerate(reader):
        if idx != 0:
            if topic_dict[row[4]] == 12:
                continue
            else:
                resampled.append(row)
                topic_dict[row[4]] += 1

# Randomly determine the order of posts in the chosen subsample
random.shuffle(resampled)

# record the original indices and the 8-digit IDs identifying each post
orig_indices = {}
indices = []
for comment in resampled:
    indices.append(int(comment[1]))
    orig_indices[int(comment[0])] = int(comment[1])

# get the contributions of various topics to each post from disk
# (requires theta_distributions-all-monthly-f from the output path of the
# relevant model)
topic_contrib_dists = {}
with open("theta_distributions-all-monthly-f","r") as thetas:
    for line in thetas:
        if line.strip() != "":
            line = line.strip().split()
            if int(line[0]) in orig_indices.keys():
                if int(line[0]) not in topic_contrib_dists.keys():
                    topic_contrib_dists[int(line[0])] = []
                topic_contrib_dists[int(line[0])].append(tuple([int(line[1]),float(line[2])]))

# store the topic contributions for the subsample on disk for easier future use
with open("topic_dist.csv","w") as csvfile:
    writer_R = csv.writer(csvfile) # initialize the CSV writer
    writer_R.writerow(['index','random_index','topic_contrib']) # write headers to the CSV file
    for comment in topic_contrib_dists.keys():
        writer_R.writerow([comment,orig_indices[comment],topic_contrib_dists[comment]])

# retrieve the text of comments in the subsample
all_indices = []
all_comments = []
with open("sampled_comments-f",'r') as original_sample:
    for index,line in enumerate(original_sample):
        if index % 2 == 0:
            all_indices.append(int(line.strip().replace("index: ","")))
        elif index % 2 != 0:
            all_comments.append(line.strip())

# associate each with the relevant 8-digit ID
zipped_all_comm = zip(all_indices,all_comments)

cleaned_up = []
for comment in zipped_all_comm:
    if comment[0] in indices:
        cleaned_up.append(comment)

# prepare to store the sets of comments assigned to each participant
remaining_indices = [x for x in indices]
participant_sets = {}

# Assign 86 unique comments to each of the 6 participants to rate
for ss in range(6):
    participant_sets[ss] = remaining_indices[(ss*86):((ss+1)*86)]
    print(len(participant_sets[ss]))
for ss in range(6):
    for element in participant_sets[ss]:
        remaining_indices.pop(remaining_indices.index(element))

# create a counter for the number of comments assigned to each participant
comment_counter = {key:0 for key in range(6)}

# divide comments not already assigned into six mutually-exclusive groups
remaining_indices_sets = numpy.array_split(numpy.array(remaining_indices),6)

# Assign each of the 6 groups simultaneously to two different raters for
# reliability evaluation purposes
sets = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]
for idx,element in enumerate(sets):
    for index in remaining_indices_sets[idx]:
        participant_sets[element[0]] = numpy.append(participant_sets[element[0]],index)
        participant_sets[element[1]] = numpy.append(participant_sets[element[1]],index)

# create two files for each participant, one including the 8-digit IDs followed
# by posts, the other a spreadsheet for recording attitude and evidence-responsiveness
for i in range(6):
    with open("cleaned_sample_ratings-"+str(i)+".csv","w+") as subset:
        writer_R = csv.writer(subset) # initialize the CSV writer
        writer_R.writerow(['index','for/against','evidence-responsiveness']) # write headers to the CSV file
    with open("cleaned_sample_ratings-"+str(i)+".csv","a+") as subset, open("cleaned_sample_posts-"+str(i),"w+") as sample:
        writer_R = csv.writer(subset) # initialize the CSV writer
        for index in participant_sets[i]:
            print(index,end="\n",file=sample)
            list = []
            list.append(index)
            writer_R.writerow(list)
            list = []
            for element in cleaned_up:
                if int(element[0]) == index:
                    print(element[1], end="\n",file=sample)
