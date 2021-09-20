import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import Error
import operator

def connect_to_database(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def main():
    DB_FILE = 'data/reddit_50_both_inferred.db'

    # get connection with database
    conn = connect_to_database(DB_FILE)
    cur = conn.cursor()
    conn.text_factory = lambda b: b.decode(errors = 'ignore')

    cur.execute("SELECT * FROM comments")
    comments_rows = cur.fetchall()

    error_subreddits = {}
    clausified_subreddits = {}
    error_date = {}
    clausified_date = {}
    total_clausified = 0
    total_errored = 0

    curr_doc_id = 1
    for row in comments_rows:
        cur.execute("SELECT * FROM classified_clauses where doc_id = ?", (str(curr_doc_id),))
        clause_row = cur.fetchone()
        subreddit = row[2]
        month = row[3]
        year = row[4]
        date = str(month) + '-' + str(year)

        if not clause_row:
            if subreddit in error_subreddits:
                error_subreddits[subreddit] = error_subreddits[subreddit] + 1
            else:
                error_subreddits[subreddit] = 1
            
            if date in error_date:
                error_date[date] = error_date[date] + 1
            else:
                error_date[date] = 1

            total_errored += 1
            curr_doc_id = curr_doc_id + 1
            continue

        if subreddit in clausified_subreddits:
            clausified_subreddits[subreddit] = clausified_subreddits[subreddit] + 1
        else:
            clausified_subreddits[subreddit] = 1

        if date in clausified_date:
            clausified_date[date] = clausified_date[date] + 1
        else:
            clausified_date[date] = 1
        
        total_clausified += 1
        curr_doc_id = curr_doc_id + 1
    
    # create temporal errored documents frequency graph
    tick_locs = [i*12 for i in range(13)]
    ticks = [(2008+i) for i in range(13)]

    prop_date = {}
    for key in error_date.keys():
        prop_date[key] = clausified_date[key] / (error_date[key] + clausified_date[key])

    plt.plot(prop_date.keys(), prop_date.values())
    plt.title('Proportion of comments successfully\nsegmented over time', fontsize=18)
    plt.xticks(tick_locs,ticks, rotation='vertical',fontsize=14)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Proportion', fontsize=18)
    plt.tight_layout()
    plt.show()

    # create temporal successful documents frequency graph
    tick_locs = [i*12 for i in range(13)]
    ticks = [(2008+i) for i in range(13)]

    # plt.plot(clausified_date.keys(), clausified_date.values())
    # plt.title('No. of segmented comments', fontsize=20)
    # plt.xticks(tick_locs,ticks, rotation='vertical',fontsize=14)
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Number of docs', fontsize=18)
    # plt.tight_layout()
    # plt.show()

    # clausified_subreddits = (dict(sorted(clausified_subreddits.items(), key=operator.itemgetter(1),reverse=True)))
    # keys = list(clausified_subreddits.keys())[:15]
    # error_vals = [error_subreddits[x] for x in keys]
    # clausified_subreddits_values = [clausified_subreddits[x] for x in keys]
    # props_success = []
    # print(error_vals)
    # print(clausified_subreddits_values)
    # for index, val in enumerate(error_vals):
    #     total = val + clausified_subreddits_values[index]
    #     props_success.append(clausified_subreddits_values[index] / total)
    
    # print(props_success)
    # labels = keys
    # x = np.arange(len(labels))
    # width = 0.4 # the width of the bars
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, props_success, width)
    # # rects2 = ax.bar(x + width/2, clausified_subreddits_values, width, label='Success')
    # ax.set_ylabel('Proportion', fontsize=14)
    # ax.set_xlabel('Subreddits', fontsize=18)
    # ax.set_title('Proportion of successfully segmented comments\nfor top 15 subreddits', fontsize=14)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, rotation=60)
    # ax.set_ylim([0.96, 1.00])

    # ax.legend()
    # # ax.bar_label(rects1, padding=6)
    # # ax.bar_label(rects2, padding=6, label_type='center')
    # fig.tight_layout()
    # plt.show()
    
    # error_subreddits = (dict(sorted(error_subreddits.items(), key=operator.itemgetter(1),reverse=True)))
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # b1 = ax.bar(list(error_subreddits.keys())[:15], list(error_subreddits.values())[:15])
    # ax.set_title('Error Subreddits')
    # ax.bar_label(b1)
    # plt.show()

    # clausified_subreddits = (dict(sorted(clausified_subreddits.items(), key=operator.itemgetter(1),reverse=True)))
    # fig2 = plt.figure()
    # ax2 = fig2.add_axes([0,0,1,1])
    # b2 = ax2.bar(list(clausified_subreddits.keys())[:15], list(clausified_subreddits.values())[:15])
    # ax2.set_title('Clausified Subreddits')
    # ax2.bar_label(b2)
    # plt.show()

if __name__ == "__main__":
    main()