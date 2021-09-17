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

lda_prep = open('lda_prep').readlines()
empty_indices = []

dict = {'doc_id': [],
        'word_topic_assignments': []
    }

df = pd.DataFrame(dict)

last_id = 300000

for i, line in enumerate(lda_prep):

    if i < last_id:
        continue

    if line == '\n':
        continue

    doc_id = i + 1
    if doc_id % 10000 == 0:
        print(doc_id)

    bow = (gensim_dictionary.doc2bow(line.split()))
    topics = model[bow][2]
    print(topics)

    df.loc[len(df.index)] = [str(doc_id), topics]
    
print(df)
df.to_pickle('new_wta.pkl')

