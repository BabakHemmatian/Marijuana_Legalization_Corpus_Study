import sqlite3
import csv
import numpy as np
from scipy.sparse import csr_matrix
import sys

num_topics = 100
print_every = 10000

indices = np.zeros([3059959,num_topics])

with open("LDA_full-corpus:True_{}/theta_distributions-all-monthly-f".format(num_topics),"r") as g:
    for line in g:
        row = line.strip().split()
        indices[int(row[0]),int(row[1])] = float(row[2])

conn = sqlite3.connect('reddit_{}.db'.format(num_topics))
cursor = conn.cursor()
cursor.execute("SELECT * from comments")
cursor.execute("ALTER TABLE comments ADD relevance integer")
cursor.execute("ALTER TABLE comments ADD attitude integer")
cursor.execute("ALTER TABLE comments ADD persuasion integer")

for i in range(num_topics):
    cursor.execute("ALTER TABLE comments ADD topic_{} real".format(i))

cursor.execute("SELECT name FROM PRAGMA_TABLE_INFO('comments')")
print(cursor.fetchall())

cursor.execute("SELECT * from comments")
for row in range(3059959):
    for element in cursor.execute('SELECT * FROM comments WHERE rowid = {}'.format(row+1)):
        for col in range(num_topics):
            if indices[row,col] != 0:
                cursor.execute("UPDATE comments SET topic_{} = {} WHERE rowid = {}".format(col,indices[row,col],row+1))

    if int(row+1) % print_every == 0:
        print(row+1)

conn.commit()

cursor.execute("SELECT * from comments")
print(cursor.fetchmany(size=100))
