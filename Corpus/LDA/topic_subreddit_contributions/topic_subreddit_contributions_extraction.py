from scipy import stats
import numpy as np
import pandas as pd
import csv
import sqlite3

conn = sqlite3.connect("reddit_50_both_inferred.db")
cursor = conn.cursor()

comment_to_count = {}
with open("data_for_R-f.csv", "r") as csv_file:
  csv_reader = csv.reader(csv_file)
  # Iterate over each row in the csv using reader object
  next(csv_reader)
  for row in csv_reader:
    # Row variable is a list that represents a row in csv
    topic_assignment = row[3]
    arr = topic_assignment[1:-1].split(",")
    # Get length of comment
    comment_to_count[int(row[0])] = len(arr)

contributions = {}
list_for_csv = []

top_subreddits = ['AskReddit', 'trees', 'politics', 'news', 'worldnews',
 'Drugs', 'Marijuana', 'The_Donald', 'todayilearned', 'Libertarian']

for subreddit in top_subreddits:
  contributions[subreddit] = [0 for _ in range(50)]

relevant_topics = [i for i in range(50)]
irrelevant_topics = [1, 20, 26, 33, 34, 39, 46]
for topic in irrelevant_topics:
    relevant_topics.remove(topic)

topic_query_list = ["topic_{}".format(i) for i in range(50)]
topic_query = ",".join(topic_query_list)
topic_query = "{}".format(topic_query)

for subreddit in top_subreddits:
  query = "SELECT ROWID, {} from comments WHERE subreddit='{}'".format(topic_query, subreddit)
  cursor.execute(query)
  comments_for_subreddit = cursor.fetchall()
  normalizing_factor = 0
  # For each comment for a particular subreddit
  for tup in comments_for_subreddit:
    row = tup[0]
    # Get length of document
    normalizing_factor += comment_to_count[row]
    # Iterate through the topic contributions in the comment
    topic_results = tup[1:]
    for topic_num, topic_contribution in enumerate(topic_results):
        if topic_num in relevant_topics:
            if topic_contribution is not None:
                # Getting number of words assigned to topic_num for this comment
                to_add = topic_contribution * comment_to_count[row]
                contributions[subreddit][topic_num] += to_add

  my_array = np.array(contributions[subreddit])
  contributions[subreddit] = my_array / normalizing_factor
  list_for_csv.append(contributions[subreddit])

with open('topic_subreddit_contributions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(list_for_csv)
conn.close()
