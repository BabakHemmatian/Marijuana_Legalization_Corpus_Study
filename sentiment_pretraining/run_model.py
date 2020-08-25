import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel

# Make Connection to Sqlite DB
conn = sqlite3.connect('reddit.db')
cursor = conn.cursor()


# Prepare data sets


# Function to transform the continuous sentiment rating into a
# discrete label
def transform_avg_sentiment(original_sentiment):
    num = float(original_sentiment)
    # Negative
    if num < 0:
        return 0
    else:
        # Neutral
        if 0 <= num <= 0.1:
            return 1
        # Positive
        else:
            return 2


def get_comment_sentiment_df():
    SQL_Query = pd.read_sql_query(
        '''select 
            original_comm,
            sentiments
            from comments
        ''', conn)
    original_df = pd.DataFrame(SQL_Query, columns=['original_comm', 'sentiments'])
    renamed_df = original_df.rename(columns={'original_comm': 'text'})
    renamed_df['labels'] = renamed_df.apply(lambda row: transform_avg_sentiment(row.sentiments), axis=1)
    return renamed_df


def create_hist_from_data(sentiments_list):
    plt.hist(sentiments_list)
    plt.show()


# Prepare data frames for training
data = get_comment_sentiment_df()[['text', 'labels']]
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Run model
model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, use_cuda=False,
                                         args={'fp16': False, 'num_train_epochs': 1, 'manual_seed':1,
                                               "eval_batch_size": 8,
                                               "train_batch_size": 8})

# Train the model
model.train_model(train)

# Evaluate model
result, model_outputs, wrong_predictions = model.eval_model(test)
