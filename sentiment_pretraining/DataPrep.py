import csv
import defaults as df
from reddit_parser import Parser
from os import path
import sentiment_pretraining.sentiment_defaults as sent_df


# Function to transform the continuous sentiment rating into a
# discrete label
def transform_avg_sentiment(original_sentiment):
    num = float(original_sentiment)
    # Negative
    if num <= -0.5:
        return 0
    else:
        # Neutral
        if -0.5 < num < 0.5:
            return 1
        # Positive
        else:
            return 2


class DataPrep(object):
    def __init__(self):
        self.parser = Parser()

    # Function to create training or testing data as a csv file with two
    # columns : comment and label.
    def create_data_set(self, file_name):
        with open(file_name, mode='w+') as file:
            field_names = ['text', 'labels']
            csv_writer = csv.DictWriter(file, fieldnames=field_names)
            csv_writer.writeheader()
            size = sent_df.train_size
            if file_name == 'test.csv':
                size = sent_df.eval_size
            for year_num in range(0, size):
                for month in df.months:
                    year = df.years[year_num]
                    fns = self.parser.get_parser_fns(year, month)
                    original_comments = fns['original_comm']
                    sentiments = fns['sentiments']
                    if not path.exists(original_comments):
                        print("Data from year " + str(year)
                              + " and month " + str(month) + " is still missing")
                    else:
                        with open(original_comments, mode='r') as comments:
                            with open(sentiments, mode='r') as sents:
                                comment = comments.readline()
                                sentiment = sents.readline()
                                while comment and sentiment:
                                    rounded = transform_avg_sentiment(sentiment)
                                    csv_writer.writerow({'text': comment, 'labels': int(rounded)})
                                    comment = comments.readline()
                                    sentiment = sents.readline()

    # Function to create data sets if they don't exist
    def prep_data(self):
        if not path.exists('train.csv') and not path.exists('test.csv'):
            print("Creating training and test sets")
            self.create_data_set('train.csv')
            self.create_data_set('test.csv')
        else:
            print("Training and test sets have already been created!"
                  " Delete the files and re-run if you would like to update them.")
