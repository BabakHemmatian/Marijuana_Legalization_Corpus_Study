# import modules
import gensim
import os
from nltk.corpus import stopwords
import csv
import pandas as pd
import string


# get the absolute path, needed for gensim loading
file_path = os.path.abspath(__file__)
path = os.path.dirname(file_path)+"/"

# load the model, dictionary and indexed corpus from disk (can be found in the shared drive)
model = gensim.models.ldamodel.LdaModel.load(path+"RC_LDA_50_True.lda")
gensim_dictionary = gensim.corpora.dictionary.Dictionary.load(path+"RC_LDA_Dict_True.dict")
# train_corpus_load = gensim.corpora.MmCorpus("RC_LDA_Corpus_True.mm")
# eval_corpus_load = gensim.corpora.MmCorpus("RC_LDA_Eval_True.mm")
# train_indices = open('LDA_train_set_True').readlines()
# eval_indices = open('LDA_eval_set_True').readlines()

lda_prep = open('lda_prep').readlines()
empty_indices = []

# dict = {'doc_id': [],
#         'word_topic_assignments': []
#     }

# df = pd.DataFrame(dict)

# saved_wta = pd.read_pickle('new_wta.pkl')
last_id = 300000

# csv_file = open('wta_new.csv', mode='a+')
# csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for i, line in enumerate(lda_prep):

    if i < last_id:
        continue

    if line == '\n':
        continue

    doc_id = i + 1
    if doc_id % 10000 == 0:
        print(doc_id)
    # if doc_id % 200000 == 0:
        # df.to_pickle('new_wta.pkl')

    bow = (gensim_dictionary.doc2bow(line.split()))
    topics = model[bow][2]
    print(topics)
    break
    # csv_writer.writerow([str(doc_id), topics])
    # df.loc[len(df.index)] = [str(doc_id), topics]
    
# print(df)
# df.to_pickle('new_wta.pkl')

# # with open('word_topic_assignments.csv', mode='rb') as f:
# #     lines =f.readlines()
# #     for line in lines:
# #         if b'word,frequency,per-topic phi\r\r\n' in line:
# #             print('boi')
    
    
# # Write the per-word topics to a CSV file
# # NOTE: in counting the alreadcy finished lines, be mindful of the blank rows and adjust 
# # the code not to start from scratch (<100k docs remaining)
# #with open("word_topic_assignments_new.csv", "a+", newline='') as f:

# # prepare the writer object and add headers
# # writer = csv.writer(f)
# #writer.writerow(["word","frequency","per-topic phi"])

# dict = {'doc_id': [],
#         'word_topic_assignments': []
#     }

# df = pd.DataFrame(dict)

# # df = pd.read_pickle('../wta.pkl')

# for id_,line in enumerate(train_corpus_load): # loop through comments

#     if id_ > 3028343:
#     #print out progress
#         #if (id_ % 10000) == 0:
#         print(id_, train_indices[id_])
#             # df.to_csv('word_topic_assignments_new.csv', index=False, mode='a+', header=False)
#             # df = pd.DataFrame(dict)
        
#         topics = model[line][2] # extract the per-word topics for the comment from the model

#         # append the word, its frequency in the document and its probability under topics to a list
#         temp = [] 
#         for word in topics:
#             for index in line:
#                 if index[0] == word[0]:
#                     count = index[1]
#             temp.append([gensim_dictionary[word[0]],count,word[1]]) 

#         df.loc[len(df.index)] = [train_indices[id_].rstrip('\n'), temp]
#         # print(df)
#         # df.loc[len(df.index)] = str(temp)
#         # print(df)
        

# print(df)
# df.to_pickle('wta_remaining.pkl')
#     #writer.writerow(temp) # print list to the csv file

# # for id_,line in enumerate(eval_corpus_load): # loop through comments

# #     #print out progress
# #     if (id_ % 10000) == 0:
# #         print(id_, eval_indices[id_])
# #         df.to_csv('word_topic_assignments_eval.csv', index=False, mode='a+', header=False)
# #         df = pd.DataFrame(dict)
    
# #     topics = model[line][2] # extract the per-word topics for the comment from the model

# #     # append the word, its frequency in the document and its probability under topics to a list
# #     temp = [] 
# #     for word in topics:
# #         for index in line:
# #             if index[0] == word[0]:
# #                 count = index[1]
# #         temp.append([gensim_dictionary[word[0]],count,word[1]]) 
    
# #     df.loc[len(df.index)] = [eval_indices[id_].rstrip('\n'), temp]

# #     #writer.writerow(temp) # print list to the csv file

