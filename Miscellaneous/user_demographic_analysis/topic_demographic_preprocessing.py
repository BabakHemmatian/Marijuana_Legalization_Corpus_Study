import csv
import os
import xlrd
import sqlite3

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

county = False
demo = xlrd.open_workbook('U.S. Religion Census Religious Congregations and Membership Study, 2010 (County File).XLSX')
demo_data = demo.sheet_by_index(0)
pol = open("2016_US_County_Level_Presidential_Results.csv", "r", encoding="latin-1")
pol_data = pol.readlines()

conn = sqlite3.connect("reddit_50_both_inferred.db")
cursor = conn.cursor()

topic_array = ["topic_{}".format(i) for i in range(0, 50)]

topic_query = ",".join(topic_array)

query = "SELECT ROWID, author, {} from comments".format(topic_query)
cursor.execute(query)
query_results = cursor.fetchall()

# Index variables for each author's list
count_index = 0
country_index = 1
state_index = 2
county_index = 3
# stores comment count, state, county in a list for each author
author_data = {}

# Iterate through all the location files
for root, dirs, files in os.walk("location"):
    for file in files:
        if file.endswith(".csv"):
            f = open("location/" + file, "r")
            file_data = f.readlines()
            for i in range(len(file_data)):
                if i > 0:
                    row = file_data[i].split(",")
                    # store comment count, country, state, and county for each author
                    author = row[0].rstrip()
                    count = row[3]
                    country = row[7].rstrip()
                    county = row[5]
                    state = row[6]
                    if not county:
                        county = state
                    author_data[author] = [int(count), country, state, county]

# populate poll data
poll_data_dict = {}
for j in range(len(pol_data)):
    if j > 0:
        pol_row = pol_data[j].split(",")
        per_GOP = float(pol_row[5])
        state_abbrev = pol_row[8]
        county = pol_row[9]
        if state_abbrev not in poll_data_dict:
            poll_data_dict[state_abbrev] = {county: per_GOP}
        else:
            poll_data_dict[state_abbrev][county] = per_GOP

# populate num words for each comment
num_words_data = {}
num_words_count = 0
with open("data_for_R-f.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        num_words_count += 1
        word_list = row['topic_assignments']
        num_words = len(eval(word_list))
        num_words_data[i] = num_words

# loop over all topics
for topic in range(0, 50):
    # create a file
    file_name = str(topic) + "_results.txt"
    results = open(file_name, "a")
    topic_contributions = []
    demographic_values = []
    count = 0
    # loop over all comments and check if comment has current topic
    for i, comment in enumerate(query_results):
        current_author = comment[1].rstrip()
        curr_perc = comment[topic + 2]
        if current_author != "[deleted]" and curr_perc is not None:
            curr_perc = float(curr_perc)
            # find author location, pass if fewer than 10 or not US
            if current_author in author_data and author_data[current_author][count_index] >= 10 and author_data[current_author][country_index] == 'US':
                county = author_data[current_author][county_index]
                state = author_data[current_author][state_index]
                row = 2
                while (row <= 3149 and demo_data.cell(row, 563).value != state and demo_data.cell(row, 566).value != county):
                    row += 1
                row_rel = row
                if (row == 3150):
                    continue
                party_data = 0
                # Poll data
                state_abbrev = us_state_abbrev[state]
                if state_abbrev in poll_data_dict and county in poll_data_dict[state_abbrev]:
                    party_data = poll_data_dict[state_abbrev][county]
                # populate the lists
                # weight topic contribution by number of words in comment
                num_words = num_words_data[i]
                curr_perc = num_words * curr_perc
                rel = demo_data.cell(row_rel, 2).value
                res = str(curr_perc) + "  " + str(rel) + "  " + str(party_data) + "\n"
                results.write(res)
                count += 1
                print(count)
        f.close()
    results.close()
    pol.close()
    conn.close()
