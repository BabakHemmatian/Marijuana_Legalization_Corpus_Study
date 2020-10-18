import torch
import os
import sqlite3
import pandas as pd
from defaults import model_path

# Establish configurable variables
# Labeling
NEGATIVE_BOUND = 0
NEUTRAL_BOUND = 0.1
NEGATIVE_LABEL = 0
NEUTRAL_LABEL = 1
POSITIVE_LABEL = 2
TEXT_COL = 'text'
LABEL_COL = 'labels'

# Batch range. Depending on your RAM constraints, you may make this
# window smaller or larger
STARTING_COMMENT = 0
ENDING_COMMENT = 50

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
EVAL_BATCH_SIZE = 100
TRAIN_BATCH_SIZE = 100
EPOCHS = 1
MANUAL_SEED = 1
SAVE_STEPS_FREQ = 1000
NUM_LABELS = 3
MODEL_TO_TRAIN = 'roberta-base'
MODEL_TO_TEST = model_path + '{}/sentiment_pretraining/outputs/'.format(model_path)
OUTPUT_DIR = model_path + '{}/sentiment_pretraining/sentiment_pretraining_steps/{}-{}'.format(model_path,
                                                                                              STARTING_COMMENT,
                                                                                              ENDING_COMMENT)
USE_CUDA = torch.cuda.is_available()
if not os.path.exists("{}/sentiment_pretraining".format(model_path)):
    os.mkdir("{}/sentiment_pretraining".format(model_path))
# Data paths
DATABASE_PATH = '{}/reddit.db'.format(model_path)
DATA_PATH = '{}/sentiment_pretraining/sentiment_pretraining_data.csv'.format(model_path)
METRICS = '{}/sentiment_pretraining/metrics.txt'.format(model_path)
WRONG_PREDICTIONS = '{}/sentiment_pretraining/wrong_predictions.csv'.format(model_path)
MODEL_OUTPUTS = "{}/sentiment_pretraining/model_outputs.txt".format(model_path)
RESULTS = "{}/sentiment_pretraining/results.txt".format(model_path)

# Evaluation
TRUE_POSITIVE = 'tp'
FALSE_POSITIVE = 'fp'
TRUE_NEGATIVE = 'tn'
FALSE_NEGATIVE = 'fn'


# Function to transform the continuous sentiment rating into a
# discrete label.
def transform_avg_sentiment(original_sentiment):
    num = float(original_sentiment)
    # Negative
    if num < NEGATIVE_BOUND:
        return NEGATIVE_LABEL
    else:
        # Neutral
        if NEGATIVE_BOUND <= num <= NEUTRAL_BOUND:
            return NEUTRAL_LABEL
        # Positive
        else:
            return POSITIVE_LABEL


# Function to create a histogram of the data to
# fine tune the label boundaries
def create_hist_from_data(sentiments_list):
    plt.hist(sentiments_list)
    plt.show()


# Function to return the training data as a dataframe
# checks if a CSV file has been saved to disk. Otherwise, it queries
# the database
def get_comment_sentiment_df():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    # If file does not exist, then read it in from the database
    conn = sqlite3.connect(DATABASE_PATH)
    SQL_Query = pd.read_sql_query(
        '''select 
        original_comm,
        sentiments
        from comments
        ''', conn)

    original_df = pd.DataFrame(SQL_Query, columns=['original_comm', 'sentiments'])
    renamed_df = original_df.rename(columns={'original_comm': 'text'})
    renamed_df[LABEL_COL] = renamed_df.apply(lambda row: transform_avg_sentiment(row.sentiments), axis=1)
    renamed_df.to_csv(DATA_PATH)
    return renamed_df
