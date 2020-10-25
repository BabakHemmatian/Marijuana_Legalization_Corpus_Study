import sqlite3
import numpy as np

conn = sqlite3.connect('reddit.db')

sql_create_comments_table = """ CREATE TABLE IF NOT EXISTS comments (
                                        original_comm text,
                                        original_indices integer,
                                        subreddit text,
                                        month integer,
                                        year integer,
                                        average_comment real,
                                        processed_text text,
                                        author_id integer,
                                        FOREIGN KEY(author_id) REFERENCES author(rowid)
                                    ); """

sql_create_sentiment_table = """ CREATE TABLE IF NOT EXISTS sentiments (
                                        month integer,
                                        year integer,
                                        start_index integer,
                                        end_index integer,
                                        average_comment,
                                        v_sentiment real,
                                        t_sentiment real,
                                        comment_id int,
                                        FOREIGN KEY (comment_id) REFERENCES comments (rowid)
                                    ); """

sql_create_author_table = """ CREATE TABLE IF NOT EXISTS author (
                                        id integer PRIMARY KEY,
                                        name text,
                                    ); """


def insert_one_month_to_authors(conn, year, month):
    c = conn.cursor()
    with open(f"author/author-{year}-{month}") as f:
        values = f.readlines()
        values = [(x.strip("\n"),) for x in values if x != '[deleted]']
        c.executemany('INSERT INTO author VALUES(?);', values);
    conn.commit()


def insert_all_to_authors(conn, year_range):
    sql_create_author_table = " CREATE TABLE IF NOT EXISTS author (name text); "
    c = conn.cursor()
    c.execute(sql_create_author_table)

    for year in year_range:
        if year == 2019:
            month_range = range(1, 10)
        else:
            month_range = range(1, 13)

        for month in month_range:
            insert_one_month_to_authors(conn, year, month)


def insert_one_month_to_comments(conn, year, month, cols):
    col_2_values = {}
    for col in cols:
        file = open(f"{col}/{col}-{year}-{month}", 'r')
        values = file.readlines()
        col_2_values[col] = values

    n = len(col_2_values[cols[0]])
    for col in cols:
        assert len(col_2_values[
                       col]) == n, f"files not aligned: the length of {col} is {len(col)} and the length of {cols[0]} is {len(cols[0])}"

    rows = []
    for i in range(n):
        row = [year, month]
        for col in cols:
            row.append(col_2_values[col][i])
        rows.append(row)

    c = conn.cursor()

    col_names = str(tuple(['year', 'month'] + cols)).replace("'", "")
    question_marks = str(tuple(['?'] * (len(cols) + 2))).replace("'", "")
    c.executemany(f"INSERT INTO comments {col_names} VALUES{question_marks};", rows);
    conn.commit()


def insert_all_to_comments(conn, year_range, cols):
    sql_create_comments_table = """ CREATE TABLE IF NOT EXISTS comments (
                                        original_comm text,
                                        original_indices integer,
                                        subreddit text,
                                        month integer,
                                        year integer,
                                        average_comment real,
                                        t_sentiments text,
                                        v_sentiments text,
                                        processed_text text,
                                        votes int,
                                        author text
                                    ); """
    c = conn.cursor()
    c.execute(sql_create_comments_table)

    for year in year_range:

        month_range = range(1, 13)

        for month in month_range:
            insert_one_month_to_comments(conn, year, month, cols)





if __name__ == "__main__":
    # choose the directories/columns that you want to insert
    cols = [
        'original_comm',
        'subreddit',
        'original_indices',
        't_sentiments',
        'v_sentiments',
        'votes',
        'author'
    ]
    # the directory names must be exactly like the column names
    insert_all_to_comments(conn, range(2008, 2020), cols=cols)
