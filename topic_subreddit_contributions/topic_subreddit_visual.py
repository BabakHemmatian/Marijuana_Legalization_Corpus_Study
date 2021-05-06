import numpy as np
import csv
import sqlite3
import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

relevant_topics = [i for i in range(50)]
irrelevant_topics = [1, 20, 26, 33, 34, 39, 46]
for topic in irrelevant_topics:
    relevant_topics.remove(topic)

top_subreddits = ['AskReddit', 'trees', 'politics', 'news', 'worldnews',
 'Drugs', 'Marijuana', 'The_Donald', 'todayilearned', 'Libertarian']


topic_names = ["Effects/prohibition vis-a-vis alcohol/tobacco",
"UNCLEAR",
"Hemp: legality and uses",
"Party politics and ideology",
"Quantities and limits",
"Legal status and common use",
"Govt. power and ind. rights/freedoms",
"Border, organized crime & Latin influence",
"State vs. federal legalization",
"Media campaigns & portrayals",
"enforcement vis-a-vis violent crimes",
"Legal market & economic forces",
"Addiction potential & gateway status",
"Reasoning and arguments",
"State-level legal. timelines",
"Police car searches",
"Medical marijuana effects and access",
"Cannabis types and use methods",
"Marijuana use and the workplace",
"FDA schedules",
"UNCLEAR",
"Continuity of US drug & foreign policy",
"Age and familial relations",
"Marijuana and finances",
"User stereotypes and life outcomes",
"Private interests & the prison industry",
"UNCLEAR",
"Legalization across US and the world",
"Police house searches & seizure",
"Legal procedures",
"Emotional and life impact",
"Reddit moderation",
"Everyday enforcement encounters",
"UNCLEAR",
"UNCLEAR",
"Drug testing",
"Judgments of character",
"Imprisonment over marijuana",
"Electoral politics & parties",
"UNCLEAR",
"Local/state regulations",
"Health and opinion research",
"DUI effects & enforcement",
"Racial/minority disparities",
"Federal Court Processes",
"Smoking methods, health effects and bans",
"UNCLEAR",
"Enforcement & observance",
"Gun versus marijuana regulations",
"Expletives-laden discourse"]

df = pd.read_csv('topic_subreddit_contributions.csv')
df = pd.DataFrame(df).T

for i, topic_name in enumerate(topic_names):
    if topic_name != "UNCLEAR":
        plt.figure()
        plt.bar(top_subreddits, df.iloc[i + 1])
        plt.ylim(bottom=0, top=100000000)
        plt.title(topic_name)
        plt.xlabel("Subreddit")
        plt.ylabel("Contribution")
plt.show()
