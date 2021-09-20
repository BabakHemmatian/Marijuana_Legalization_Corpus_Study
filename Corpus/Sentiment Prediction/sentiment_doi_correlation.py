import pandas as pd
from scipy.stats import pointbiserialr
import sqlite3

conn = sqlite3.connect("reddit_50_both_inferred.db")
cursor = conn.cursor()

# Get sentiment results and human ratings
query = "SELECT t_sentiments, v_sentiments, sentiments, attitude, persuasion FROM comments WHERE attitude IS NOT NULL AND persuasion IS NOT NULL"

cursor.execute(query)

results = cursor.fetchall()

t_sentiments_average = []
v_sentiments_average = []
sentiments = []
attitude = []
persuasion = []

for tup in results:
    # Get list of human ratings for each of the comments
    attitude_array = str(tup[3]).split(",")
    persuasion_array = str(tup[4]).split(",")

    # Iterate through the ratings and add them to a list
    for index in range(0, min(len(attitude_array), len(persuasion_array))):
        attitude.append(int(attitude_array[index]))
        persuasion.append(int(persuasion_array[index]))

        # Compute average of the Text Blob and Vader Sentiment values for a comment
        t_sentiments = [float(i) for i in tup[0].split(",") if i != '']
        v_sentiments = [float(i) for i in tup[1].split(",") if i != '']

        # Store everything
        t_sentiments_average.append(sum(t_sentiments)/ len(t_sentiments))
        v_sentiments_average.append(sum(v_sentiments)/ len(v_sentiments))
        sentiments.append(float(tup[2]))

# Print results
print("PBC of t_sentiments and attitude (human)")
print(pointbiserialr(t_sentiments_average, attitude))

print("PBC of v_sentiments and attitude (human)")
print(pointbiserialr(v_sentiments_average, attitude))

print("PBC of avg_sentiments and attitude (human)")
print(pointbiserialr(sentiments, attitude))

print("PBC of t_sentiments and persuasion (human)")
print(pointbiserialr(t_sentiments_average, persuasion))

print("PBC of v_sentiments and persuasion (human)")
print(pointbiserialr(v_sentiments_average, persuasion))

print("PBC of avg_sentiments and persuasion (human)")
print(pointbiserialr(sentiments, persuasion))

conn.close()
