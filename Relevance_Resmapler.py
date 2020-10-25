import csv
import sys
import random
from config import *
import numpy

per_participant = int(rel_sample_num / num_annot) # number of posts to assign to each annotator
unique_participant = int(per_participant - (per_participant * overlap))

# load the random sample of documents
resampled = []
rand_id = []
with open(model_path + "/auto_labels/sample_labeled-{}-{}.csv".format(rel_sample_num,balanced_rel_sample),'r+') as csvfile:
    reader = csv.reader(csvfile)
    # read human data for sampled comments one by one
    for idx,row in enumerate(reader):
        if idx != 0:
            if len(row) != 0:
                resampled.append(row)
                random_id = random.randint(10000000,99999999)
                while random_id in rand_id:
                    random_id = random.randint(10000000,99999999)
                rand_id.append(random_id)
                resampled[-1].append(random_id)

# Randomly determine the order of posts in the sample
remaining_indices = [x[-1] for x in resampled]
numpy.random.shuffle(resampled)

participant_sets = {}
# Assign unique comments to each of the participants to rate
for ss in range(num_annot):
    participant_sets[ss] = remaining_indices[(ss*unique_participant):((ss+1)*unique_participant)]
for ss in range(num_annot):
    for element in participant_sets[ss]:
        remaining_indices.pop(remaining_indices.index(element))

# create a counter for the number of comments assigned to each participant
comment_counter = {key:0 for key in range(num_annot)}

# divide comments not already assigned into mutually-exclusive groups
remaining_indices_sets = numpy.array_split(numpy.array(remaining_indices),num_annot*2)

# Assign each of the groups simultaneously to two different raters for
# reliability evaluation purposes
sets = [[0,1],[1,2],[0,2]]
for idx,element in enumerate(sets):
    for index in remaining_indices_sets[idx]:
        participant_sets[element[0]] = numpy.append(participant_sets[element[0]],index)
        participant_sets[element[1]] = numpy.append(participant_sets[element[1]],index)

# create two files for each participant, one including the 8-digit IDs followed
# by posts, the other a spreadsheet for recording attitude and evidence-responsiveness
for i in range(num_annot):
    with open(model_path+"/auto_labels/rel_sample_info-{}-{}-{}.csv".format(per_participant,balanced_rel_sample,i),"w+") as subset:
        writer_R = csv.writer(subset) # initialize the CSV writer
        writer_R.writerow(['year', 'month', 'text', 'auto label','index','accuracy','attitude','argumentation']) # write headers to the CSV file
    with open(model_path+"/auto_labels/rel_sample_info-{}-{}-{}.csv".format(per_participant,balanced_rel_sample,i),"a+") as subset:
        writer_R = csv.writer(subset) # initialize the CSV writer
        for element in resampled:
            if element[-1] in participant_sets[i]:
                writer_R.writerow(element)
    with open(model_path+"/auto_labels/rel_sample_ratings-{}-{}-{}.csv".format(per_participant,balanced_rel_sample,i),"w+") as ratings:
        writer_R = csv.writer(ratings)
        writer_R.writerow(['index','text','relevance','attitude','argumentation'])
    with open(model_path+"/auto_labels/rel_sample_ratings-{}-{}-{}.csv".format(per_participant,balanced_rel_sample,i),"a+") as ratings:
        writer_R = csv.writer(ratings)
        for element in resampled:
            if element[-1] in participant_sets[i]:
                writer_R.writerow([element[-1],element[2]])
