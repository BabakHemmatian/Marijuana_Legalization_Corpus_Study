The contents of this folder can be used to get clause classifications for genericity, aspect, and boundedness. 

Follow these steps - 

0. Create your virtual environment with the requirements.txt in this folder. 

1. Call clausify_batch.py with the segmented batch file as the argument. This will generate a new clausified batch file that would not contain any EDU_BREAKS. This also does some post-processing like remove noisy lines.

2. Call classifier.py with the clausified file as the argument. This will classify all the clauses in the batch and create a pickle file with the following columns - doc_id, clauses, genericity_preds, genericity_softmax, aspect_preds, aspect_softmax, boundedness_preds, boundedness_softmax, ne_tags.