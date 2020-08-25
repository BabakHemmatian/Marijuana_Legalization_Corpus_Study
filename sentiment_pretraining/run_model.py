import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Make Connection to Sqlite DB
conn = sqlite3.connect('reddit.db')
cursor = conn.cursor()


# Prepare data sets


# Function to transform the continuous sentiment rating into a
# discrete label
def transform_avg_sentiment(original_sentiment):
    num = float(original_sentiment)
    # Negative
    if num < -0.1:
        return 0
    else:
        # Neutral
        if -0.1 <= num <= 0.1:
            return 1
        # Positive
        else:
            return 2


def get_comment_sentiment_df():
    SQL_Query = pd.read_sql_query(
        '''select 
            original_comm,
            t_sentiments,
            v_sentiments
            from comments
        ''', conn)
    original_df = pd.DataFrame(SQL_Query, columns=['original_comm', 't_sentiments',
                                                   'v_sentiments'])
    original_df['avg_sentiment'] = (np.add(original_df['t_sentiments'].split(","),
                                           original_df['v_sentiments'].split(","))) \
                                   / len(original_df['v_sentiments'].split(","))
    return original_df


def create_hist_from_data():
    sentiments_list = get_comment_sentiment_df()['avg_sentiment'].to_list()
    plt.hist(sentiments_list)
    plt.show()



create_hist_from_data()
# Prepare data frames for training

# train_df = pd.read_csv('train.csv', dtype={'labels': 'int64'})

# eval_df = pd.read_csv('test.csv', dtype={'labels': 'int64'})


# Run model
# model = Model(train_df, eval_df)

# model.train()

# Evaluate model
# result, model_outputs, wrong_predictions = model.eval()
