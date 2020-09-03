import csv
import numpy as np
from sentiment_pretraining_defaults import *
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel

# Get data sets
data = get_comment_sentiment_df()[[TEXT_COL, LABEL_COL]]
train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

freqs = {0: 0, 1: 0, 2: 0}

for post in train.itertuples():
    freqs[post.labels] += 1

print("Original training class frequencies: ")
print(freqs)

test_freqs = {0: 0, 1: 0, 2: 0}

# Evaluate Model
for post in test.itertuples():
    test_freqs[post.labels] += 1
print("Original test class frequences: ")
print(test_freqs)

# Create model
to_test_model = ClassificationModel('roberta', MODEL_TO_TEST, num_labels=NUM_LABELS, use_cuda=USE_CUDA,
                                    args={'fp16': False, 'num_train_epochs': EPOCHS, 'manual_seed': MANUAL_SEED,
                                          'save_steps': SAVE_STEPS_FREQ, 'save_optimizer_and_scheduler': True,
                                          "eval_batch_size": EVAL_BATCH_SIZE,
                                          "train_batch_size": TRAIN_BATCH_SIZE})

result, model_outputs, wrong_predictions = to_test_model.eval_model(test, acc=accuracy_score)

# Write f1, accuracy, and mcc metrics
result_file = open(METRICS, "a")
result_file.write(str(result))
result_file.close()

# Write wrong predictions
with open(WRONG_PREDICTIONS, 'a') as file:
    field_names = [TEXT_COL, LABEL_COL]
    csv_writer = csv.DictWriter(file, fieldnames=field_names)
    csv_writer.writeheader()
    for example in wrong_predictions:
        csv_writer.writerow({TEXT_COL: example.text_a, LABEL_COL: example.label})

# Write outputs of the model
with open(MODEL_OUTPUTS, "a") as output_file:
    for output in model_outputs:
        output_file.write(str(output) + "\n")

labels = np.zeros([len(eval_df), 1])
predictions = np.zeros([len(eval_df), 1])

for i in range(0, len(eval_df)):
    labels[i] = eval_df.iloc[i].labels
    predictions[i] = np.argmax(model_outputs[i])
labels = labels.flatten()
predictions = predictions.flatten()

label_measures = []
for value in [0, 1, 2]:
    label_measures.append(dict())

for label in [0, 1, 2]:
    for index, ind_label in enumerate(labels):
        if labels[index] == label and labels[index] == predictions[index]:
            if TRUE_POSITIVE in label_measures[label]:
                label_measures[label][TRUE_POSITIVE] += 1
            else:
                label_measures[label][TRUE_POSITIVE] = 1
        elif labels[index] == label:
            if FALSE_NEGATIVE in label_measures[label]:
                label_measures[label][FALSE_NEGATIVE] += 1
            else:
                label_measures[label][FALSE_NEGATIVE] = 1
        elif labels[index] != label and predictions[index] != label:
            if TRUE_NEGATIVE in label_measures[label]:
                label_measures[label][TRUE_NEGATIVE] += 1
            else:
                label_measures[label][TRUE_NEGATIVE] = 1
        elif labels[index] != label and predictions[index] == label:
            if FALSE_POSITIVE in label_measures[label]:
                label_measures[label][FALSE_POSITIVE] += 1
            else:
                label_measures[label][FALSE_POSITIVE] = 1

print(label_measures)

precisions = []
recalls = []
f1s = []
for label in [0, 1, 2]:
    try:
        tp = label_measures[label][TRUE_POSITIVE]
    except IndexError:
        tp = 0
    try:
        fp = label_measures[label][FALSE_POSITIVE]
    except IndexError:
        fp = 0
    try:
        fn = label_measures[label][FALSE_NEGATIVE]
    except IndexError:
        fn = 0

    if tp != 0 or fp != 0:
        precisions.append(float(tp) / (float(tp) + float(fp)))

        with open(RESULTS, "a+") as f:
            f.write("precisions: " + str(precisions) + "\n")

    if tp != 0 or fn != 0:
        recalls.append(float(tp) / (float(tp) + float(fn)))

        with open(RESULTS, "a+") as f:
            f.write("recalls: " + str(recalls) + "\n")

    if tp != 0 or (fn != 0 and fp != 0):
        f1s.append(2 * (precisions[label] * recalls[label]) / (precisions[label] + recalls[label]))

        with open(RESULTS, "a+") as f:
            f.write("f1s: " + str(f1s) + "\n")
