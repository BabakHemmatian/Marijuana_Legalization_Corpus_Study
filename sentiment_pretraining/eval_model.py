from simpletransformers.classification import ClassificationModel
import csv
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import os
import pdb;
pdb.set_trace()


# Load Trained Model

model = ClassificationModel('bert', 'outputs/', num_labels=3, use_cuda=False,
                            args={'fp16': False, 'num_train_epochs': 2, 'manual_seed': 1,
                                  "eval_batch_size": 8,
                                  "train_batch_size": 8})

train_df = pd.read_csv('train.csv', dtype={'labels': 'int64'})
freqs = {0: 0, 1: 0, 2: 0}

for post in train_df.itertuples():
    freqs[post.labels] += 1

print("Original training class frequencies: ")
print(freqs)

test_freqs = {0: 0, 1: 0, 2: 0}

# Evaluate Model
eval_df = pd.read_csv('test.csv', dtype={'labels': 'int64'})
for post in eval_df.itertuples():
    test_freqs[post.labels] += 1
print("Original test class frequences: ")
print(test_freqs)

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=accuracy_score)

# Write f1, accuracy, and mcc metrics
result_file = open("sentiment_pretraining/metrics.txt", "a")
result_file.write(str(result))
result_file.close()
# Write wrong predictions
with open('sentiment_pretraining/wrong_predictions.csv', 'a') as file:
    field_names = ['text', 'labels']
    csv_writer = csv.DictWriter(file, fieldnames=field_names)
    csv_writer.writeheader()
    for example in wrong_predictions:
        csv_writer.writerow({'text': example.text_a, 'labels': example.label})
# Write outputs of the model

with open("sentiment_pretraining/model_outputs.txt", "a") as output_file:
    for output in model_outputs:
        output_file.write(str(output) + "\n")

labels = np.zeros([len(eval_df), 1])
preds = np.zeros([len(eval_df), 1])
for i in range(0, len(eval_df)):
    labels[i] = eval_df.iloc[i].labels
    preds[i] = np.argmax(model_outputs[i])
labels = labels.flatten()
preds = preds.flatten()
label_measures = []
for value in [0, 1, 2]:
    label_measures.append(dict())

for label in [0, 1, 2]:
    for index, ind_label in enumerate(labels):
        if labels[index] == label and labels[index] == preds[index]:
            if 'tp' in label_measures[label]:
                label_measures[label]['tp'] += 1
            else:
                label_measures[label]['tp'] = 1
        elif labels[index] == label:
            if 'fn' in label_measures[label]:
                label_measures[label]['fn'] += 1
            else:
                label_measures[label]['fn'] = 1
        elif labels[index] != label and preds[index] != label:
            if 'tn' in label_measures[label]:
                label_measures[label]['tn'] += 1
            else:
                label_measures[label]['tn'] = 1
        elif labels[index] != label and preds[index] == label:
            if 'fp' in label_measures[label]:
                label_measures[label]['fp'] += 1
            else:
                label_measures[label]['fp'] = 1

print(label_measures)

precisions = []
recalls = []
f1s = []
for label in [0, 1, 2]:
    try:
        tp = label_measures[label]['tp']
    except:
        tp = 0
    try:
        fp = label_measures[label]['fp']
    except:
        fp = 0
    try:
        fn = label_measures[label]['fn']
    except:
        fn = 0

    if tp != 0 or fp != 0:
        precisions.append(float(tp) / (float(tp) + float(fp)))

        with open("sentiment_pretraining/results.txt", "a+") as f:
            f.write("precisions: " + str(precisions) + "\n")

    if tp != 0 or fn != 0:
        recalls.append(float(tp) / (float(tp) + float(fn)))

        with open("sentiment_pretraining/results.txt", "a+") as f:
            f.write("recalls: " + str(recalls) + "\n")

    if tp != 0 or (fn != 0 and fp != 0):
        f1s.append(2 * (precisions[label] * recalls[label]) / (precisions[label] + recalls[label]))

        with open("sentiment_pretraining/results.txt", "a+") as f:
            f.write("f1s: " + str(f1s) + "\n")
