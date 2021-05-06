from collections import Counter, namedtuple, defaultdict
import numpy as np
import pandas as pd
import csv
import sqlite3
from csv import reader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tsmoothie.smoother import *


conn = sqlite3.connect("reddit_50_both_inferred.db")
cursor = conn.cursor()
subreddits = ['AskReddit', 'trees', 'politics', 'news', 'worldnews', 'Drugs', 'Marijuana', 'The_Donald', 'todayilearned', 'Libertarian']
query = "SELECT month, year, inferred_attitude FROM comments WHERE inferred_attitude IS NOT NULL AND subreddit = 'Libertarian'"
headers = [("Proportion of Positive Comments Over Time, Subreddit: r/{}", "Proportion of Positive Comments"), ("Number of Positive Comments Over Time, Subreddit: r/{}", "Number of Positive Comments")]
for header in headers:
    for subreddit in subreddits:
        query = "SELECT month, year, inferred_attitude FROM comments WHERE inferred_attitude IS NOT NULL AND subreddit='{}'".format(subreddit)
        cursor.execute(query)
        results = cursor.fetchall()
        results_dict = {}
        average_attitude_dict = {}

        for tup in results:
            if tup[1] in results_dict:
                if tup[0] in results_dict[tup[1]]:
                    results_dict[tup[1]][tup[0]].append(tup[2])
                else:
                    results_dict[tup[1]][tup[0]] = []
            else:
                average_attitude_dict[tup[1]] = {}
                results_dict[tup[1]] = {}

        for dict in results_dict:
            for inner_dict in results_dict[dict]:
                if len(results_dict[dict][inner_dict]) > 0:
                    if "Proportion" in header[0]:
                        average_attitude_dict[dict][inner_dict] = sum(results_dict[dict][inner_dict]) / len(results_dict[dict][inner_dict])
                    else:
                        average_attitude_dict[dict][inner_dict] = sum(results_dict[dict][inner_dict])
        years = range(2008, 2020)

        months = [1, 6, 12]

        values = []
        x_axis = []
        x_axis_visual = []
        count = 0
        for year in years:
            if year in average_attitude_dict:
                for month in months:
                    if month in average_attitude_dict[year]:
                        x_axis_visual.append("(" + str(year) + "," + str(month) + ")")
                        x_axis.append(count)
                        count += 1
                        values.append(average_attitude_dict[year][month])
        plt.figure()
        smoother = LowessSmoother(smooth_fraction=0.4, iterations=1)
        smoother.smooth(values)
        low_pi, up_pi = smoother.get_intervals('prediction_interval')
        low_ci, up_ci = smoother.get_intervals('confidence_interval')
        plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
        plt.plot(smoother.data[0], '.k')
        plt.fill_between(range(len(smoother.data[0])), low_pi[0], up_pi[0], alpha=0.3, color='blue')
        plt.fill_between(range(len(smoother.data[0])), low_ci[0], up_ci[0], alpha=0.3, color='blue')
        plt.xlabel('Date')
        plt.title(header[0].format(subreddit))
        plt.ylabel(header[1])
        plt.xticks(x_axis, x_axis_visual, rotation=90)
plt.show()
conn.close()
