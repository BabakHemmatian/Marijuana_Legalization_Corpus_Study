from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import csv
import fnmatch
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

### hyperparameters
ep = 1 # number of epochs
k = 10 # number of folds
trial = 1
full = False # If True, the model will be trained on all available data.
# If False, models will be trained on subsets and tested using cross-validation.
weight = [1,1] # class weights

# Output paths based on parameters. Currently based on number of epochs and trial
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path) + "/"+str(ep)+"_"+str(trial)+"/"

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

### Since the human-rated posts were predominantly relevant, de-duplicated
# negative examples are sampled here from those LDA topics whose top comments
# were mostly irrelevant for more balanced training
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
# its various iterations--are added here to the sample. Babak Hemmatian rated
# the relevance of posts in these samples
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


### aggregate the samples above into simpletransformers' default formate:
# a list of lists where each inner list has two elements: first, the text of the
# comment as a string. Second, its integer label(0 if irrelevant, 1 if relevant)
# NOTE: Posts about which raters disagreed are considered relevant.
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

### Training and evaluation
if not full: # if training and evaluating using cross-validation

    kfold = KFold(k, True, 1) # # Set up the k folds, shuffle, and use 1 as seed
    k = 0 # fold counter

    # containers for cross-folds evaluation results
    f1_folds = []
    precision_folds = []
    recall_folds = []

    for train, test in kfold.split(data): # for each training and eval fold

        ## create containers for the data
        train_data = []
        eval_data = []
        ## populate the containers based on the fold indices
        for idx,post in enumerate(data):
            if idx in train:
                train_data.append(post)
            elif idx in test:
                eval_data.append(post)
        if k == 0: # for the first fold
            print("Size of the training fold: ")
            print(len(train_data))
            print("Size of the evaluation fold: ")
            print(len(eval_data))

        k += 1 # update the fold counter
        output_path = path + "/"+str(k)+"/" # path for storing model and results

        ## set up neural network runtime configurations
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        ## Datasets need to be in Pandas Dataframes
        train_df = pd.DataFrame(train_data)
        eval_df = pd.DataFrame(eval_data)

        ## Define the classification model object. Model configurations changed
        # by adding the correct entries to the "args" dict. See simpletransformers
        # on github for more information

        model = ClassificationModel('roberta', 'roberta-base', use_cuda=False,
        weight = weight, args={'fp16': False,'num_train_epochs': ep, 'manual_seed':1})

        ## Train the model
        model.train_model(train_df, output_dir=output_path)

        ## Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df, accuracy=accuracy_score)

        ### write model results to file.
        # NOTE: results.txt saves f1, precision, recall and accuracy for a fold
        # NOTE: wrong.txt saves evaluation set documents that were wrongly
        # classified, along with the model's prediction for each
        # NOTE: Model_outputs saves the output layer's activation for various
        # classes for each evaluation set document
        with open(output_path+'results.txt', 'w') as f, open(output_path+'wrong.txt','w') as g:
            f.write(''.join('{}:{}\n'.format(key, val) for key, val in result.items()))
            for prediction in wrong_predictions:
                g.write(prediction.text_a + "; " + str(prediction.label))
                g.write("\n")
        np.save(output_path+"model_outputs",model_outputs)

        # calculate precision, recall and f1
        precision = (float(result['tp']) / (float(result['tp'])+float(result['fp'])))
        recall = (float(result['tp']) / (float(result['tp'])+float(result['fn'])))
        f1 = (2 * (precision*recall))/(precision + recall)

        # write results to file
        with open(output_path+"results.txt","a+") as f:
            f.write("precision: "+ str(precision) + "\n")
            f.write("recall: "+ str(recall) + "\n")
            f.write("f1: "+ str(f1) + "\n")

        # adds the fold results to lists
        precision_folds.append(precision)
        recall_folds.append(recall)
        f1_folds.append(f1)


    # Writes fold micro-averages and their macro average to file
    with open("cross_folds_res.txt","r") as full_results:
        full_results.write("Fold f1: "+str(f1_folds))
        full_results.write("Mean f1: "+str(np.mean(f1_folds)))
        full_results.write("Folds precision: "+str(precision_folds))
        full_results.write("Mean precision: "+str(np.mean(precision_folds)))
        full_results.write("Fold recall: "+str(recall_folds))
        full_results.write("Mean recall: "+str(np.mean(recall_folds)))

else: # if training on all available data

    ## path currently set to number of data points
    output_path = path + "/full_" + str(len(data)) + "/"

    ## set up neural network runtime configurations
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    data = pd.DataFrame(data) # turn the data into a Pandas dataframe

    ## Define the classification model object. See above for more details
    model = ClassificationModel('roberta', 'roberta-base', num_labels=3, use_cuda=False,
    args={'fp16': False,'num_train_epochs': ep, 'manual_seed':1})

    ## Train the model
    model.train_model(data, output_dir=output_path)
