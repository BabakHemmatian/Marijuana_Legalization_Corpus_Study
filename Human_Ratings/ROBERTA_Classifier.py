from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import csv
import fnmatch
import os
import sys
import numpy as np
from pathlib2 import Path

### hyperparameters
epochs = 1
training_fraction = 0.9
# add other hyperparameters as needed

# TODO: set up k-fold cross-validation

### This section loads ratings of 6 human raters for comments subsampled from
# posts representative of LDA topics. Topics whose top posts were mostly composed
# of irrelevant posts were previously excluded from sampling
post_filenames=[]
rating_filenames=[]
for file in os.listdir('.'):
    if fnmatch.fnmatch(file, 'cleaned_sample_posts-*'):
        post_filenames.append(file)
    elif fnmatch.fnmatch(file, 'cleaned_sample_ratings-*'):
        rating_filenames.append(file)

posts=[]
idxs=[]
for file in post_filenames:
    with open(file,'r') as f:
        for idx,line in enumerate(f):
            if idx % 2 != 0:
                posts.append(line.strip())
            else:
                idxs.append(int(line.strip()))
combined = {}
for index,post in enumerate(posts):
    combined[idxs[index]] = post

human_ratings = [] # initialize counter for the number of valid human ratings
for rater_id,file in enumerate(rating_filenames):
    with open(file,'r') as csvfile:
        reader = csv.reader(csvfile)
        # read human data for sampled comments one by one
        for idx,row in enumerate(reader):
            # ignore headers
            if ( idx != 0):
                if row[1] == "":
                    human_ratings.append((int(row[0]),0,rater_id))
                else:
                    human_ratings.append((int(row[0]),1,rater_id))

### 20% of the posts were rated by two raters to evaluate inter-rater agreement.
# here, we identify those posts for which raters gave disparate relevance ratings
not_matching = []
for post in human_ratings:
    id = post[0]
    rating = post[1]
    for other_posts in human_ratings:
        if other_posts[0] == id and other_posts[1] != rating:
            not_matching.append(int(id))

# TODO: calculate Cohen's Kappa for pairs of raters

### Since the human-rated posts were predominantly relevant, de-duplicated
# negative examples are sampled here from those LDA topics whose top comments
# were mostly irrelevant
with open('sample_keys-f.csv','r') as lda_ratings:
    reader = csv.reader(lda_ratings)
    for idx, row in enumerate(reader):
        cnt = 0
        if (idx != 0) and (row[6].strip() != "") and row[6].isnumeric():
            if int(row[6]) > 10:
                continue
            elif int(row[1]) not in combined.keys():

                with open('sampled_comments-f','r') as lda_posts:
                    for line in lda_posts:
                        if cnt == 1:
                            combined[int(row[1])] = line
                            human_ratings.append((int(row[1]),0,6))
                            cnt = 0
                        if row[1] in line:
                            cnt = 1
        elif (idx != 0) and row[6].strip() != "":
            if int(row[1]) not in combined.keys():

                with open('sampled_comments-f','r') as lda_posts:
                    for line in lda_posts:
                        if cnt == 1:
                            combined[int(row[1])] = line
                            human_ratings.append((int(row[1]),0,6))
                            cnt = 0
                        if row[1] in line:
                            cnt = 1

### To further balance the training set, negative examples from random samples
# of 2008, 2013 and 2018 posts--extracted to evaluate and improve the regex across
# its various iterations are added here to the sample. Babak Hemmatian personally
# rated the relevance of posts in these samples
rand_filenames=[]
for file in os.listdir('.'):
    if 'sampled_' in file and 'csv' in file:
        rand_filenames.append(file)

print(rand_filenames)

for file in rand_filenames:
    with open(file,"r") as rand_sample:
        reader = csv.reader(rand_sample)
        for idx,row in enumerate(reader):
            if idx != 0 and row[1] == '0':
                # give a random unused index
                proposed = np.random.randint(10000000,100000000)
                while proposed in combined.keys():
                    proposed = np.random.randint(10000000,100000000)
                human_ratings.append((proposed,0,6))
                combined[proposed] = row[2]


### aggregate the samples into a list of lists where each inner list has two
# elements: first, the text of the comment as a string. Second, its integer label
# (0 if irrelevant, 1 if relevant)
# NOTE: Posts about which raters disagreed are considered relevant
data = []
sampled_so_far = []
for post in human_ratings:
    if post[0] not in not_matching and post[0] not in sampled_so_far:
        data.append([combined[post[0]],post[1]])
        sampled_so_far.append(post[0])
    elif post[0] in not_matching:
        data.append([combined[post[0]],1])

### examine sample balance
sum = 0
for elem in data:
    if elem[1] == 1:
        sum += 1
print("number of positive examples: "+str(sum))
print("number of negative examples: "+str(len(data)-sum))

### define the training and evaluation sets
train_size = int(training_fraction * len(data))
val_size = len(data) - train_size

# Divide the dataset by randomly selecting samples.
train_data = []
eval_data = []
eval_indices = np.random.choice(len(data),val_size)

# populate the sets
for idx,post in enumerate(data):
    if idx in eval_indices:
        eval_data.append(post)
    else:
        train_data.append(post)

### set up neural network runtime configurations
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Datasets need to be in Pandas Dataframes
train_df = pd.DataFrame(train_data)
eval_df = pd.DataFrame(eval_data)

### Define the classification model object. Model configurations can be changed
# by adding the correct entries to the "args" dict. See simpletransformers' github
# page for more information
# TODO: Allow loading of previously checkpointed models
model = ClassificationModel('roberta', 'roberta-base',use_cuda=False,args={'fp16': False,'num_train_epochs': epochs, 'manual_seed':0, 'use_early_stopping':True,'save_eval_checkpoints':True}) # You can set class weights by using the optional weight argument

### Train the model
model.train_model(train_df)

### Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

### write model results to file
# TODO: write models with different hyperparameters to different folders
with open('results.txt', 'w') as f, open('wrong.txt','w') as g:
    f.write(''.join('{}:{}\n'.format(key, val) for key, val in result.items()))
    np.save("output",model_outputs)
    g.write('\n'.join('{}'.format(prediction) for prediction in wrong_predictions))
