'''
LABELS 
Genericity - GENERIC: 0 ; SPECIFIC: 1
Aspect - DYNAMIC: 0 ; STATIVE: 1
Boundedness - BOUNDED:0 ; UNBOUNDED: 1
CANNOT_DECIDE: 2
'''


# IMPORTS
import sys
import os
from transformers import RobertaTokenizer, TFRobertaModel
import spacy
import tokenizations
from numpy import asarray
from numpy import savetxt, loadtxt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import xml.etree.ElementTree as ET 
from keras import backend as K
from sklearn.model_selection import train_test_split
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sutime import SUTime
import json
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

# GLOBAL VARIABLES

POS = True 
NE = True

MAX_CLAUSE_LENGTH = 70

TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")  
MODEL = TFRobertaModel.from_pretrained('roberta-base')

# Spacy and corenlp stuff 
nlp = spacy.load("en_core_web_sm")

## METRIC FUNCTIONS #####################################################

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

########################################################################

def get_temporal_info(clause):
  ''' 
  Given a clause, this function gets a one hot representation of the temporal
  annotations in the clause. There are 4 types. DATE, TIME, DURATION, SET
  '''

  annotations = sutime.parse(clause)
  types = {'DATE':0, 'TIME':0, 'DURATION':0, 'SET':0}
  for annotation in annotations:
    if annotation['type'] == 'DATE':
      types['DATE'] = 1
    elif annotation['type'] == 'TIME':
      types['TIME'] = 1
    elif annotation['type'] == 'DURATION':
      types['DURATION'] = 1
    elif annotation['type'] == 'SET':
      types['SET'] = 1
    
  return tf.cast(tf.stack(list(types.values())), dtype='float32')

def get_one_hot_pos(pos):
  '''
  Given a POS tag, return its one hot encoding that will be concatenated to the
  word embedding
  @param pos: the POS tag
  @return: one one hot encoding of the POS tag
  '''
  POS_TAGS = {
      'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CONJ': 4, 'CCONJ': 5, 'DET': 6,
        'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
        'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, 'SPACE': 18
  }
  one_hot_matrix = tf.eye(len(POS_TAGS))
  if pos == '':
    return tf.zeros(len(POS_TAGS))
  return one_hot_matrix[POS_TAGS[pos]]

def get_one_hot_ne(ne):
  '''
  Given a NE tag, return its one hot encoding that will be concatenated to the
  word embedding
  @param NE: the NE tag
  @return: one one hot encoding of the NE tag
  '''
  NE_TAGS = { 'PERSON': 0, 'NORP': 1, 'FAC': 2, 'ORG': 3, 'GPE': 4,'LOC': 5,
                  'PRODUCT': 6, 'EVENT': 7, 'WORK_OF_ART': 8, 'LAW': 9, 
                  'LANGUAGE': 10, 'DATE': 11, 'TIME': 12, 'PERCENT': 13, 
                  'MONEY': 14, 'QUANTITY': 15, 'ORDINAL': 16, 'CARDINAL': 17
  }
  one_hot_matrix = tf.eye(len(NE_TAGS))
  if ne == '':
    return tf.zeros(len(NE_TAGS))
  return one_hot_matrix[NE_TAGS[ne]]


def add_pos_ne_encoding(tokens, doc, vectors, pos=True, ne=True):
  '''
  This function takes in the tokens for a clause, the nlp doc with the tags, and the vectors.
  It concatenates the POS tag encoding to their respective word vectors.
  @param tokens:
  @param doc:
  @param vectors:
  @return: 
  '''
  # for each token, find its POS tag, get encoding, concatenate to its vector.
  # if the token doesn't have a tag, pad with 0s 

  if (not pos) and (not ne):
    return vectors

  spacy_tokens = [token.text for token in doc]
  roberta_tokens = tokens 
  a2b, b2a = tokenizations.get_alignments(spacy_tokens, roberta_tokens)

  new_vectors = []
  for index, alignment in enumerate(b2a):
    if alignment:
      # get the tags from that spacy token and concat
      named_entity_tags = doc[alignment[0]].ent_type_
      pos_tags = doc[alignment[0]].pos_
      new_vectors.append(tf.concat([vectors[0][index], get_one_hot_pos(pos_tags), get_one_hot_ne(named_entity_tags)], axis=0)) 
    else:
      # concat zeros 
      new_vectors.append(tf.concat([vectors[0][index], tf.zeros([37])], 0))  

  new_vectors = tf.stack(new_vectors)
  return new_vectors

def get_entity_types(doc):
  clause_ne = []
  for e in doc.ents:
    clause_ne.append((str(e.start_char), e.text, e.label_))
  return clause_ne

def make_embeddings(clause, tokenizer, model, pos=True, ne=True):
  doc = nlp(clause)

  # tokenize and get the output from the model
  tokenized_clause = tokenizer(clause, padding='max_length', max_length=MAX_CLAUSE_LENGTH, return_tensors='tf')
  try:
    outputs = model(**tokenized_clause)
  except:
    return -1, -1, -1
  # get the tokens from the tokenizer
  tokens = (tokenizer.convert_ids_to_tokens(tokenized_clause['input_ids'][0]))

  # add POS and NE encodings 
  vectors_with_pos_ne = add_pos_ne_encoding(tokens, doc, outputs.last_hidden_state, pos, ne)

  clause_entity_type_tuples = get_entity_types(doc)

  return vectors_with_pos_ne, clause_entity_type_tuples, 0


##############################
# MAIN
##############################

def main(file_path):
  '''
  @param file_path: path to the clausified file
  '''
  
  GENERICITY_MODEL_PATH = ''
  BOUNDEDNESS_MODEL_PATH = ''
  ASPECT_MODEL_PATH = ''

  clausified_file = open(file_path)
  file_id = os.path.basename(file_path).split('.txt')[0].split('_')[-1]

  data = {'doc_id':[], 'clauses':[], 'genericity_pred':[], 
    'genericity_softmax':[], 'aspect_pred':[], 'aspect_softmax':[],
    'boundedness_pred':[], 'boundedness_softmax':[], 'ne_tags':[]}

  # check last classified doc in the batch
  try:
    pickled_data = pd.read_pickle(r'./batch_' + file_id + r'.pkl')
    last_doc_id_saved = pickled_data.doc_id.iat[-1]
    while True:
      clause = clausified_file.readline()
      if 'DOC_BREAK' in clause:
        curr_doc_id = clause.split()[1]
        if int(curr_doc_id) == int(last_doc_id_saved):
          break
    
    data = pickled_data.to_dict('list')
  except:
    print('No pickled file already saved, new one will be created.')

  genericity_model = load_model(GENERICITY_MODEL_PATH, compile = True, custom_objects={'f1_m':f1_m,'recall_m':recall_m,'precision_m':precision_m})
  aspect_model = load_model(ASPECT_MODEL_PATH, compile = True, custom_objects={'f1_m':f1_m,'recall_m':recall_m,'precision_m':precision_m})
  boundedness_model = load_model(BOUNDEDNESS_MODEL_PATH, compile = True, custom_objects={'f1_m':f1_m,'recall_m':recall_m,'precision_m':precision_m})

  genericity_labels = {0:'Generic', 1:'Specific', 2:'Unclear'}
  aspect_labels = {0:'Dynamic', 1:'Stative', 2:'Unclear'}
  boundedness_labels = {0:'Bounded', 1:'Unbounded'}

  clauses_features = []
  raw_clauses = []
  clauses_ne_tuples = []
  skip = 0

  while True:

    # Get next line from file
    clause = clausified_file.readline().rstrip()

    if 'DOC_BREAK' in clause:
      # get the id of the document
      doc_id = clause.split()[1]
      
      # if skip is set to 1
      if skip:
        skip = 0
        clauses_features = []
        raw_clauses = []
        clauses_ne_tuples = []
        continue

      # stack the features for the clauses in the doc, continue if failed because of clauses length more than 70
      try:
        clauses_input = tf.stack(clauses_features)
      except:
        clauses_features = []
        raw_clauses = []
        clauses_ne_tuples = []
        continue

      # get predictions and softmax for genericity and aspect 
      genericity_softmax = genericity_model.predict(clauses_input)
      genericity_preds = tf.math.argmax(genericity_softmax, axis=1)

      aspect_softmax = aspect_model.predict(clauses_input)
      aspect_preds = tf.math.argmax(aspect_softmax, axis=1)

      boundedness_softmax = boundedness_model.predict(clauses_input)
      boundedness_preds = tf.math.argmax(boundedness_softmax, axis=1)

      # concat temporal features 
      # get predictions and softmax for boundedness

      data['doc_id'].append(str(doc_id))
      #print(*raw_clauses, sep = "\n")
      data['clauses'].append('\n'.join(raw_clauses))
      # print(*(genericity_preds.numpy()), sep=',')
      data['genericity_pred'].append(','.join(genericity_preds.numpy().astype('<U6').tolist()))
      # print(*(aspect_preds.numpy()), sep=',')
      data['aspect_pred'].append(','.join(aspect_preds.numpy().astype('<U6').tolist()))
      # print(*(boundedness_preds.numpy()), sep=',')
      data['boundedness_pred'].append(','.join(boundedness_preds.numpy().astype('<U6').tolist()))

      # adding ne tags 
      doc_ne_tags = []
      for clause in clauses_ne_tuples:
        clause_ne_strings = []
        for ne_tuple in clause:
          tuple_string = '(' + ','.join(ne_tuple) + ')' 
          clause_ne_strings.append(tuple_string)       

        clause_ne_string = ','.join(clause_ne_strings) 
        doc_ne_tags.append(clause_ne_string)
      data['ne_tags'].append('\n'.join(doc_ne_tags))

      # getting softmax strings
      genericity_softmax = genericity_softmax.astype('<U6').tolist()
      doc_genericity_softmax = []
      aspect_softmax = aspect_softmax.astype('<U6').tolist()
      doc_aspect_softmax = []
      boundedness_softmax = boundedness_softmax.astype('<U6').tolist() 
      doc_boundedness_softmax = []    
      for index, clause_gen_softmax in enumerate(genericity_softmax):
        doc_genericity_softmax.append('(' + ','.join(clause_gen_softmax) + ')')
        doc_aspect_softmax.append('(' + ','.join(aspect_softmax[index]) + ')')
        doc_boundedness_softmax.append('(' + ','.join(boundedness_softmax[index]) + ')')

      data['genericity_softmax'].append(','.join(doc_genericity_softmax))
      data['aspect_softmax'].append(','.join(doc_aspect_softmax))
      data['boundedness_softmax'].append(','.join(doc_boundedness_softmax))

      clauses_features = []
      raw_clauses = []
      clauses_ne_tuples = []

      if int(doc_id) % 1000 == 0:
        pd.DataFrame(data).to_pickle("./batch_" + file_id +".pkl")

      continue

    raw_clauses.append(clause)
    clause_features, clause_ne_tuples, errored = make_embeddings(clause, TOKENIZER, MODEL, pos=True, ne=True)
    if errored == -1:
      skip = 1
    else:
      clauses_features.append(clause_features)
      clauses_ne_tuples.append(clause_ne_tuples)

    # if line is empty
    # end of file is reached
    if clause == 'END_OF_BATCH':
        print('huh')
        break

  pd.DataFrame(data).to_pickle("./batch_" + file_id + ".pkl")

##########################################################################################

if __name__ == "__main__":

  if len(sys.argv) < 2:
        print('ERROR: Provide file path! \nUsage: python classifier.py [FILE_PATH]')
        sys.exit(2)

  file_path = sys.argv[1]

  main(file_path)