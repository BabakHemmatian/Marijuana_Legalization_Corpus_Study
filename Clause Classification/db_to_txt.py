import sqlite3
import re

conn = sqlite3.connect('/users/asriva11/data/bhemmati/marijuana_study/reddit_50_both_inferred.db')

cursor = conn.execute("SELECT original_comm from comments")

counter = 0

for row in cursor:
	print('parsing row ' + str(counter) + '...')
	text = row[0].decode('utf-8','ignore').encode("utf-8")

	text = re.sub('\[(.+)\] {0,1}\((https?:\/\/[^\s]+)?(?: "(.+)")?\)', r'\1' ,text)	
    text = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', text)

	text = text.replace('.)', ')')
	text = text.replace('.(', '(')
	text = text.replace(').', ')')
    text = text.replace('(.', '(')
    text = text.replace('.', ' . ')
    text = text.replace('?', ' ? ')
    text = text.replace('!', ' ! ')
    text = text.replace('"', ' " ')
    text = text.replace('’', '\'')

	filename = './comments/' +  str(counter) + '.txt'
	f = open(filename, "w")
	f.write(text)
	f.close()
	counter +=1