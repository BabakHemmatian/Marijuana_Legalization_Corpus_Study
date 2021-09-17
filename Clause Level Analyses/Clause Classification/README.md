# Clause Classification

The contents of this folder can be used to get clause classifications for genericity, aspect, and boundedness. 

Follow these steps - 

0. Create your virtual environment with the requirements.txt in this folder. 

1. Call clausify_batch.py with the segmented batch file as the argument. This will generate a new clausified batch file that would not contain any EDU_BREAKS. This also does some post-processing like remove noisy lines.

2. Call classifier.py with the clausified file as the argument. This will classify all the clauses in the batch and create a pickle file with the following columns - doc_id, clauses, genericity_preds, genericity_softmax, aspect_preds, aspect_softmax, boundedness_preds, boundedness_softmax, ne_tags. You will need to provide the path to the saved models in this file. 

## Files and directories in this directory - 

1. batch_classifier.py - The file called by the batch job on CCV's OSCAR. It first clausifies (cleans the output of the segmentation tool to create clause) and then 
2. classifier.py - Given a batch of clauses, this file generates a pickle file storing the predicted classifications (genericity, aspect, and boundedness), softmax values, and recognized named entities. 
3. clausify_batch.py - Given the output of the segmentation algorithm, this file clausifies it (puts each clause on a new line). 
4. db_to_txt.py - This files gets the comments from the comments table in the database and saves them as .txt files. 
