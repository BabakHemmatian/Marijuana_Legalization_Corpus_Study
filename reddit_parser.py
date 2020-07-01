import bz2
import copy
import errno
import lzma
import zstandard as zstd
from langdetect import DetectorFactory
from langdetect import detect
from collections import defaultdict, OrderedDict
import datetime
import hashlib
import html
import json
import multiprocessing
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from math import floor,ceil
import os
import io
from pathlib2 import Path
import pickle
import re
import time
import subprocess
from pycorenlp import StanfordCoreNLP
import sys
from textblob import TextBlob
from config import *
from Utils import *
from transformers import BertTokenizer
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import fnmatch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from keras.preprocessing.sequence import pad_sequences
import hashlib
import csv
import shutil
import ahocorasick

### Wrapper for the multi-processing parser

# NOTE: This needs to be importable from the main module for multiprocessing
# https://stackoverflow.com/questions/24728084/why-does-this-implementation-of-multiprocessing-pool-not-work

def parse_one_month_wrapper(args):
    year, month, on_file, kwargs = args
    Parser(**kwargs).parse_one_month(year, month)

### Create global helper function for formatting names of data files

## Format dates to be consistent with pushshift file names
def format_date(yr, mo):
    if len(str(mo)) < 2:
        mo = '0{}'.format(mo)
    assert len(str(yr)) == 4
    assert len(str(mo)) == 2
    return "{}-{}".format(yr, mo)


## Raw Reddit data filename format. The compression types for dates handcoded
# based on https://files.pushshift.io/reddit/comments/

# IMPORTANT: Remember to recode the filenames below accordingly if the RC files
# have been downloaded and transformed into a different compression type than
# shown to prevent filename errors

def get_rc_filename(yr, mo):
    date = format_date(yr, mo)
    if (yr == 2017 and mo == 12) or (yr == 2018 and mo < 10):
        return 'RC_{}.xz'.format(date)
    elif (yr == 2018 and mo >= 10) or (yr > 2018):
        return 'RC_{}.zst'.format(date)
    else:
        return 'RC_{}.bz2'.format(date)


## based on provided dates, gather a list of months for which data is already
# available
on_file = []
for date in dates:
    mo, yr = date[0], date[1]
    proper_filename = get_rc_filename(mo, yr)
    if Path(data_path + proper_filename).is_file():
        on_file.append(proper_filename)

### Define the parser class

class Parser(object):
    # Parameters:
    #   dates: a list of (year,month) tuples for which data is to be processed
    #   path: Path for data and output files.
    #   stop: List of stopwords.
    #   vote_counting: Include number of votes per comment in parsed file.
    #   NN: Parse for neural net.
    #   write_original: Write a copy of the raw file.
    #   download_raw: If the raw data doesn't exist in path, download a copy from
    #       https://files.pushshift.io/reddit/comments/.
    #   clean_raw: Delete the raw data file when finished.

    def __init__(self, nlp_wrapper=StanfordCoreNLP('http://localhost:9000'),bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True), clean_raw=CLEAN_RAW, dates=dates,
                 download_raw=DOWNLOAD_RAW, hashsums=None, NN=NN, data_path=data_path,
                 model_path=model_path,legality=legality, marijuana=marijuana,
                 stop=stop, write_original=WRITE_ORIGINAL,array=None,
                 vote_counting=vote_counting,author=author, sentiment=sentiment,
                 add_sentiment=add_sentiment,balanced_rel_sample=balanced_rel_sample,
                 machine=None, on_file=on_file, num_process=num_process,
                 rel_sample_num=rel_sample_num, num_cores=num_cores,
                 Neural_Relevance_Filtering=Neural_Relevance_Filtering):
        # check input arguments for valid type
        assert type(vote_counting) is bool
        assert type(author) is bool
        assert type(sentiment) is bool
        assert type(add_sentiment) is bool
        assert type(NN) is bool
        assert type(write_original) is bool
        assert type(download_raw) is bool
        assert type(clean_raw) is bool
        assert type(data_path) is str
        assert type(model_path) is str
        assert type(num_cores) is int
        # check the given path
        if not os.path.exists(data_path) or not os.path.exists(model_path):
            raise Exception('Invalid path')
        assert type(stop) is set or type(stop) is list

        self.clean_raw = CLEAN_RAW
        self.dates = dates
        self.download_raw = download_raw
        self.hashsums = hashsums
        self.NN = NN
        self.data_path = data_path
        self.model_path = model_path
        self.legality = legality
        self.marijuana = marijuana
        self.stop = stop
        self.write_original = write_original
        self.vote_counting = vote_counting
        self.author = author
        self.sentiment = sentiment
        self.add_sentiment = add_sentiment
        self.num_cores = num_cores
        self.array = array
        self.machine = machine
        self.on_file = on_file
        self.bert_tokenizer = bert_tokenizer
         # connect the Python wrapper to the server
        # Instantiate CoreNLP wrapper than can be used across multiple threads
        self.nlp_wrapper = nlp_wrapper
        self.num_process = num_process
        self.rel_sample_num = rel_sample_num
        self.balanced_rel_sample = balanced_rel_sample
        self.Neural_Relevance_Filtering = Neural_Relevance_Filtering

    ## Download Reddit comment data
    def download(self, year=None, month=None, filename=None):
        assert not all([isinstance(year, type(None)),
                        isinstance(month, type(None)),
                        isinstance(filename, type(None))
                        ])
        assert isinstance(filename, type(None)) or (isinstance(year, type(None))
                                                    and isinstance(month, type(None)))
        BASE_URL = 'https://files.pushshift.io/reddit/comments/'
        if not isinstance(filename, type(None)):
            url = BASE_URL + filename
        else:
            url = BASE_URL + get_rc_filename(year, month)
        print('Sending request to {}.'.format(url))
        try:
            os.system('cd {} && wget -nv {}'.format(self.data_path, url)) # non-verbose
        except: # if download fails, mark the months affected so that they can
        # be re-downloaded
            print("Download error for year "+str(year)+", month "+str(month))
            with open(self.data_path+"Download_Errors.txt","a+") as file:
                file.write(str(year)+","+str(month)+"\n")

    ## Get Reddit compressed data file hashsums to check downloaded files'
    # integrity
    def Get_Hashsums(self):
        # notify the user
        print('Retrieving hashsums to check file integrity')
        # set the URL to download hashsums from
        url = 'https://files.pushshift.io/reddit/comments/sha256sum.txt'
        # remove any old hashsum file
        if Path(self.model_path + '/sha256sum.txt').is_file():
            os.remove(self.model_path + '/sha256sum.txt')
        # download hashsums
        os.system('cd {} && wget {}'.format(self.model_path, url))
        # retrieve the correct hashsums
        hashsums = {}
        with open(self.model_path + '/sha256sum.txt', 'rb') as f:
            for line in f:
                line = line.decode("utf-8")
                if line.strip() != "":
                    (val, key) = str(line).split()
                    hashsums[key] = val
        return hashsums

    ## calculate hashsums for downloaded files in chunks of size 4096B
    def sha256(self, fname):
        hash_sha256 = hashlib.sha256()
        with open("{}/{}".format(self.model_path, fname), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    ## Define the function for parts of preprocessing that are shared between
    # LDA and neural nets
    def _clean(self, text):

        # check input arguments for valid type
        assert type(text) is str

        replace = {"should've": "shouldve", "mustn't": "mustnt",
                   "shouldn't": "shouldnt", "couldn't": "couldnt", "shan't": "shant",
                   "needn't": "neednt", "-": ""}
        substrs = sorted(replace, key=len, reverse=True)
        regexp = re.compile('|'.join(map(re.escape, substrs)))
        stop_free = regexp.sub(
            lambda match: replace[match.group(0)], text)

        # remove special characters
        special_free = ""
        for word in stop_free.split():
            if "http" not in word and "www" not in word:  # remove links
                word = re.sub('[^A-Za-z0-9]+', ' ', word)
                if word.strip() != "":
                    special_free = special_free + " " + word.strip()

        # check for stopwords again
        special_free = " ".join([i for i in special_free.split() if i not in
                                 self.stop])

        return special_free

    ## NN_encode: uses the BERT tokenizer to process a comment into its
    ## sentence-by-sentence segment IDs and vocabulary IDs
    def NN_encode(self, text):
        # check input arguments for valid type
        assert type(text) is list or type(text) is str

        # Create 2d arrays for sentence ids and segment ids.
        sentence_ids = [] # each subarray is an array of vocab ids for each token in the sentence
        segment_ids = [] # each subarray is an array of ids indicating which sentence each token belongs to
        # The following code will:
        #   (1) Tokenize each sentence.
        #   (2) Prepend the `[CLS]` token to the start of each sentence.
        #   (3) Append the `[SEP]` token to the end of each sentence.
        #   (4) Map tokens to their IDs.
        id = 0
        for index, sent in enumerate(text):  # iterate over the sentences
            encoded_sent = self.bert_tokenizer.encode(sent,  # Sentence to encode
                                                      add_special_tokens=True)  # Add '[CLS]' and '[SEP]'
            segment = [id] * len(self.bert_tokenizer.tokenize(sent))
            sentence_ids.append(encoded_sent)
            segment_ids.append(segment)
            # # alternate segment id between 0 and 1
            # # TODO: Ask Babak about this
            id = 1 - id
        return sentence_ids, segment_ids

    ## Gets attention masks so BERT knows which tokens correspond to real words vs padding
    def NN_attention_masks(self, input_ids):
        # Create attention masks
        attention_masks = []
        for sent in input_ids:
            # Create mask.
            #   - If a token ID is 0, it's padding -- set the mask to 0.
            #   - If a token ID is > 0, it's a real token -- set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)
        return attention_masks

    ## Main parsing function for BERT
    def parse_for_bert(self, body):
        # Encode the sentences into sentence and segment ids using BERT
        sentence_ids, segment_ids = self.NN_encode(body)  # encode the text for NN
        # TODO: double check with Babak on max length
        max_length = 128
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        # Pad sentences to fit length
        padded_sentence_ids = pad_sequences(segment_ids, maxlen=max_length, dtype="long",
                                            value=0, truncating="post", padding="post")
        # Create attention masks
        attention_masks = self.NN_attention_masks(padded_sentence_ids)
        data_to_write = {
            'tokenized_sentences': body,
            'sentence_ids': padded_sentence_ids.tolist(),
            ## These below should also be ndarrays
            'segment_ids': segment_ids,
            'attention_masks': attention_masks
        }
        return data_to_write

    ## define the preprocessing function to lemmatize, and remove punctuation,
    # special characters and stopwords (LDA)

    # NOTE: Since LDA doesn't care about sentence structure, unlike NN_clean,
    # the entire comment should be fed into this function as a continuous string

    # NOTE: Quotes (marked by > in the original dataset) are not removed
    def LDA_clean(self, text):

        special_free = self._clean(text)
        # remove stopwords --> check to see if apostrophes are properly encoded
        stop_free = " ".join([i for i in special_free.lower().split() if i.lower() not
                              in self.stop])
        # load lemmatizer with automatic POS tagging
        lemmatizer = spacy.load('en', disable=['parser', 'ner'])
        # Extract the lemma for each token and join
        lemmatized = lemmatizer(stop_free)
        normalized = " ".join([token.lemma_ for token in lemmatized])
        return normalized

    ## Define the input/output paths and filenames for the parser
    def get_parser_fns(self, year=None, month=None):
        assert ((isinstance(year, type(None)) and isinstance(month, type(None))) or
                (not isinstance(year, type(None)) and not isinstance(month, type(None))))
        if isinstance(year, type(None)) and isinstance(month, type(None)):
            suffix = ""
        else:
            suffix = "-{}-{}".format(year, month)
        fns = dict((("original_comm", "{}/original_comm/original_comm{}".format(self.model_path, suffix)),
                    ("original_indices", "{}/original_indices/original_indices{}".format(self.model_path, suffix)),
                    ("counts", "{}/counts/RC_Count_List{}".format(self.model_path, suffix)),
                    ("timedict", "{}/timedict/RC_Count_Dict{}".format(self.model_path, suffix)),
                    ("total_count", "{}/total_count/total_count{}".format(self.model_path, suffix))
                    ))
        if self.NN:
            # fns["nn_prep"] = "{}/nn_prep/nn_prep{}".format(self.model_path, suffix)
            fns["bert_prep"] = "{}/bert_prep/bert_prep{}.json".format(self.model_path, suffix)
        else:
            fns["lda_prep"] = "{}/lda_prep/lda_prep{}".format(self.model_path, suffix)
        if self.vote_counting:
            fns["votes"] = "{}/votes/votes{}".format(self.model_path, suffix)
        if self.author:
            fns["author"] = "{}/author/author{}".format(self.model_path, suffix)
        if self.sentiment:
            fns["t_sentiments"] = "{}/t_sentiments/t_sentiments{}".format(self.model_path, suffix)
            fns["v_sentiments"] = "{}/v_sentiments/v_sentiments{}".format(self.model_path, suffix)
            if not self.add_sentiment:
                fns["sentiments"] = "{}/sentiments/sentiments{}".format(self.model_path, suffix)
        return fns

        ## Receives as input one document and its index, as well as output address
    # writes sentiment values derived from 2 packages separately to file,
    # averages them if add_sentiment == False, and stores the average as well

    # NOTE: If add_sentiment == True, the averaging will happen through
    # add_sentiment() within NN_Book_Keeping.py

    def write_avg_sentiment(self, original_body, month, main_counter, fns, v_sentiments=None,
                            t_sentiments=None,sentiments=None):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        tokenized = sent_detector.tokenize(original_body)

        total_vader = 0
        total_textblob = 0

        if self.machine == "local":
            with open(fns["v_sentiments"], "a+") as v_sentiments, open(fns["t_sentiments"], "a+") as t_sentiments:
                for sentence in tokenized:
                    # Vader score
                    sid = SentimentIntensityAnalyzer()
                    score_dict = sid.polarity_scores(sentence)
                    total_vader += score_dict['compound']
                    v_sentiments.write(str(score_dict['compound']) + ",")

                    # Get TextBlob sentiment
                    blob = TextBlob(sentence)
                    total_textblob += blob.sentiment[0]
                    t_sentiments.write(str(blob.sentiment[0]) + ",")
                v_sentiments.write("\n")
                t_sentiments.write("\n")

        elif self.machine == "ccv":
            v_per_sentence = []
            t_per_sentence = []
            for sentence in tokenized:
                # Vader score
                sid = SentimentIntensityAnalyzer()
                score_dict = sid.polarity_scores(sentence)
                total_vader += score_dict['compound']
                v_per_sentence.append(str(score_dict['compound']))

                # Get TextBlob sentiment
                blob = TextBlob(sentence)
                total_textblob += blob.sentiment[0]
                t_per_sentence.append(str(blob.sentiment[0]))
            v_sentiments.append(",".join(v_per_sentence))
            t_sentiments.append(",".join(t_per_sentence))

        avg_vader = total_vader / len(tokenized)
        avg_blob = total_textblob / len(tokenized)

        if not self.add_sentiment:
            avg_score = (avg_vader + avg_blob) / 2
            if self.machine == "local":
                print(avg_score, file=sentiments)
            elif self.machine == "ccv":
                sentiments.append(avg_score)

    @staticmethod
    def is_relevant(text, automaton_marijuana, automaton_legal, regex_marijuana, regex_legal):
        """
        This function determines if a given comment is relevant (it mentions both marijuana and legal topics).
        :param text: lower case text of a comment
        :param automaton_marijuana: ahocorasick.Automaton object containing the marijuana key words
        :param automaton_legal: ahocorasick.Automaton object containing the legal key words
        :param regex_marijuana: a single regular expression
        :param regex_legal: a single regular expression
        :return: Boolean, True if text is relevant, False otherwise
        """
        for _ in automaton_marijuana.iter(text):
            # note that we enter the loop only 1% of the time
            for _ in automaton_legal.iter(text):
                if not regex_marijuana.search(text) is None:
                    # if the comment is marijuana relevant, check if it legal-relevant
                    if not regex_legal.search(text) is None:
                        return True
                    else:
                        # the comment is marijuana relevant, but not legal relevant
                        return False
                else:
                    # the marijuana regex didn't match anything. So the comment is NOT relevant.
                    return False
            # the automaton_legal didn't find anything, so the comment is NOT relevant
            return False
        # the automaton_marijuana didn't find anything, so the comment is NOT relevant
        return False

    ## The main parsing function
    # NOTE: Parses for LDA if NN = False
    # NOTE: Saves the text of the non-processed comment to file as well if write_original = True
    def parse_one_month(self, year, month):
        timedict = dict()

        # get the relevant compressed data file name
        filename = get_rc_filename(year, month)

        # Get names of processing files
        fns = self.get_parser_fns(year, month)

        if self.NN or self.sentiment:  # if parsing for an NN or calculating sentiment
            # import the pre-trained PUNKT tokenizer for determining sentence boundaries
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        decoder = json.JSONDecoder()

        # check to see if fully preprocessed data for a certain month exists
        missing_parsing_files = []
        for file in fns.keys():
            if not Path(fns[file]).is_file():
                missing_parsing_files.append(fns[file])

        if len(missing_parsing_files) != 0:  # if the processed data is incpmplete

                        # this will be used for efficient word searching
            marijuana_keywords, legal_keywords = [], []
            with open("alt_marijuana.txt", 'r') as f:
                for line in f:
                    marijuana_keywords.append(line.lower().rstrip("\n"))

            with open("alt_legality.txt", 'r') as f:
                for line in f:
                    legal_keywords.append(line.lower().rstrip("\n"))

            automaton_marijuana = ahocorasick.Automaton()
            automaton_legal = ahocorasick.Automaton()

            for idx, key in enumerate(marijuana_keywords):
                automaton_marijuana.add_word(key, (idx, key))

            for idx, key in enumerate(marijuana_keywords):
                automaton_legal.add_word(key, (idx, key))

            automaton_marijuana.make_automaton()
            automaton_legal.make_automaton()

            print("The following needed processed file(s) were missing for "
                  + str(year) + ", month " + str(month) + ":")
            print(missing_parsing_files)
            print("Initiating preprocessing of " + filename + " at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))

            # preprocess raw data
            # if the file is available on disk and download is on, prevent deletion
            if not filename in self.on_file and self.download_raw:
                self.download(year, month)  # download the relevant file

                # check data file integrity and download again if needed

                # NOTE: sometimes inaccurate reported hashsums in the online dataset
                # cause this check to invariably fail. Comment out the code section
                # below if that becomes a problem.

                # calculate hashsum for the data file on disk
                filesum = self.sha256(filename)
                attempt = 0  # number of hashsum check trials for the current file
                # # if the file hashsum does not match the correct hashsum
                # while filesum != self.hashsums[filename]:
                #     attempt += 1  # update hashsum check counter
                #     if attempt == 3:  # if failed hashsum check three times,
                #     # ignore the error to prevent an infinite loop
                #         print("Failed to pass hashsum check 3 times. Ignoring.")
                #         break
                #     # notify the user
                #     print("Corrupt data file detected")
                #     print("Expected hashsum value: " +
                #           self.hashsums[filename]+"\nBut calculated: "+filesum)
                #     os.remove(self.path+'/'+filename)  # remove the corrupted file
                #     self.download(year, month)  # download it again

            # if the file is not available, but download is turned off
            elif not filename in self.on_file:
                # notify the user
                print('Can\'t find data for {}/{}. Skipping.'.format(month, year))
                return

            # create a file to write the processed text to
            if self.NN and self.machine == "local":  # if doing NN on a local computer
                fout = open(fns["bert_prep"], 'w')
            elif self.NN and self.machine == "ccv": # on a cluster
                fout = []
            elif not self.NN and self.machine == "local":  # if doing LDA on a local computer
                fout = open(fns["lda_prep"], 'w')
            elif not self.NN and self.machine == "ccv":
                fout = []
            else:
                raise Exception("Machine specification variable not found.")

            # create a file if we want to write the original comments and their indices to disk
            if self.write_original and self.machine == "local":
                foriginal = open(fns["original_comm"], 'w')
                main_indices = open(fns["original_indices"], 'w')
            elif self.write_original and self.machine == "ccv":
                foriginal = []
                main_indices = []
            elif self.write_original:
                raise Exception("Machine specification variable not found.")

            # if we want to record the votes
            if self.vote_counting and self.machine == "local":
                # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
                vote = open(fns["votes"], 'w')
            elif self.vote_counting and self.machine == "ccv":
                vote = []
            elif self.vote_counting:
                raise Exception("Machine specification variable not found.")

            # if we want to record the author
            if self.author and self.machine == "local":
                # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
                author = open(fns["author"], 'w')
            elif self.author and self.machine == "ccv":
                author = []
            elif self.author:
                raise Exception("Machine specification variable not found.")

            if self.sentiment and self.machine == "local":
                # docs for sentence-level sentiments of posts
                v_sentiments = open(fns["v_sentiments"], 'w')
                t_sentiments = open(fns["t_sentiments"], 'w')
                if not self.add_sentiment: # doc for average post sentiment
                    sentiments = open(fns["sentiments"], 'w')

            elif self.sentiment and self.machine == "ccv":
                # lists for sentence-level sentiments of posts
                v_sentiments = []
                t_sentiments = []
                if not self.add_sentiment: # list for average post sentiment
                    sentiments = []

            elif self.sentiment:
                raise Exception("Machine specification variable not found.")

            # create a file to store the relevant cummulative indices for each month
            ccount = open(fns["counts"], 'w')

            warning_counter = 0
            main_counter = 0

            # open the file as a text file, in utf8 encoding, based on encoding type
            if '.zst' in filename:
                file = open(self.data_path + filename, 'rb')
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(file)
                fin = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='ignore')
            elif '.xz' in filename:
                fin = lzma.open(self.data_path + filename, 'r')
            elif '.bz2' in filename:
                fin = bz2.BZ2File(self.data_path + filename, 'r')
            else:
                raise Exception('File format not recognized')

            # read data
            per_file_counter = 0
            for line in fin:  # for each comment
                main_counter += 1  # update the general counter

                if '.zst' not in filename:
                    line = line.decode('utf-8','ignore')

                try:
                    comment = decoder.decode(line)
                    original_body = html.unescape(comment["body"])  # original text
                except:
                    warning_counter += 1
                    if warning_counter < 10:
                        print("Warning! Invalid JSON sequence encountered. Ignoring this document.")
                        continue
                    elif warning_counter == 10:
                        print("Too many errors. Warnings turned off.")
                        continue
                    else:
                        continue

                original_body = html.unescape(comment["body"])  # original text

                is_relevant = Parser.is_relevant(original_body.lower(), automaton_marijuana, automaton_legal, marijuana[0], legality[0])

                # filter comments by relevance to the topic according to regex
                if is_relevant:

                    # preprocess the comments
                    ## TODO is this for one comment or for all of the comments in a month?
                    if self.NN:
                        # Tokenize the sentences
                        # body = sent_detector.tokenize(
                        #   original_body)
                        # Get JSON formatted objects for BERT
                        # data_to_write = self.parse_for_bert(body)
                        # Write to bert_prep folder
                        # with open(fns["bert_prep"]) as readfile:
                        #     if readfile.read(1) == "":
                        #         data = {}
                        #         data["parsed_data"] = [data_to_write]
                        #         with open(fns["bert_prep"], 'w') as outfile:
                        #             json.dump(data, outfile, indent=5)
                        #     else:
                        #         content = readfile.read()
                        #         data = json.loads(content)
                        #         temp = data['parsed_data']
                        #         temp.append(data_to_write)
                        #         data = temp
                        #         with open(fns["bert_prep"], 'w') as outfile:
                        #             json.dump(data, outfile, indent=5)

                        # If calculating sentiment, write the average sentiment to
                        # file. Range is -1 to 1, with values below 0 meaning neg
                        # sentiment.
                        # body = self._clean(original_body).lower()
                        non_url = 0
                        for word in original_body.strip().split():
                            if "http" not in word and "www" not in word:  # remove links
                                non_url += 1
                        if non_url == 0:
                            body = ""
                        else:
                            body = original_body

                    else:  # if doing LDA

                        # clean the text for LDA
                        body = self.LDA_clean(original_body)

                    if body.strip() == "":  # if the comment is not empty after preprocessing
                        pass
                    else:
                        if not self.NN:
                            # remove mid-comment lines
                            body = body.replace("\n", "")
                            body = " ".join(body.split())

                        # If calculating sentiment, write the average sentiment.
                        # Range is -1 to 1, with values below 0 meaning neg
                        # BUG: sentiment fn should be adjusted for local/ccv
                        if self.sentiment and not self.add_sentiment:
                            self.write_avg_sentiment(original_body,month,
                                                    main_counter, fns,
                                                    v_sentiments,t_sentiments,
                                                    sentiments)
                        elif self.sentiment:
                            self.write_avg_sentiment(original_body,month,
                                                    main_counter, fns,
                                                    v_sentiments,t_sentiments)

                        if self.machine == "local": # write comment-by-comment
                            # print the comment to file
                            print(body, sep=" ", end="\n", file=fout)

                            # if we want to write the original comment to disk
                            if self.write_original:
                                original_body = original_body.replace(
                                    "\n", "")  # remove mid-comment lines
                                # record the original comment
                                print(" ".join(original_body.split()), file=foriginal)
                                # record the index in the original files
                                print(main_counter, file=main_indices)

                            # if we are interested in the upvotes
                            if self.vote_counting:
                                if type(comment["score"]) is int:
                                    print(int(comment["score"]), end="\n", file=vote)
                                    # write the fuzzed number of upvotes to file
                                # Some of the scores for banned subreddits like "incels"
                                # are not available in the original dataset. Write NA for
                                # those
                                elif comment["score"] is None:
                                    print("None", end="\n", file=vote)

                            # if we are interested in the author of the posts
                            if self.author:
                                # write their username to file
                                print(comment["author"].strip(),
                                      end="\n", file=author)

                        elif self.machine == "ccv":

                            if not self.NN:
                                fout.append(body + "\n")

                            if self.write_original:
                                original_body = original_body.replace(
                                    "\n", "")  # remove mid-comment lines
                                # record the original comment
                                original_body = " ".join(original_body.split())
                                foriginal.append(original_body)
                                # record the index in the original files
                                main_indices.append(main_counter)

                            if self.vote_counting:
                                if type(comment["score"]) is int:
                                    vote.append(int(comment["score"]))
                                    # write the fuzzed number of upvotes to file
                                # Some of the scores for banned subreddits like "incels"
                                # are not available in the original dataset. Write NA for
                                # those
                                elif comment["score"] is None:
                                    vote.append("None")

                            if self.author:
                                author.append(comment["author"].strip())

                        else:
                            raise Exception("Machine identification variable not found")

                        # record the number of documents by year and month
                        created_at = datetime.datetime.fromtimestamp(
                            int(comment["created_utc"])).strftime('%Y-%m')
                        timedict[created_at] = timedict.get(created_at, 0)
                        timedict[created_at] += 1
                        per_file_counter += 1

            # write the total number of posts from the month to disk to aid in
            # calculating proportion relevant if calculate_perc_rel = True
            if calculate_perc_rel:
                with open(fns["total_count"], 'w') as counter_file:
                    print(str(main_counter), end="\n", file=counter_file)

            # close the files to save the data
            fin.close()
            if self.machine == "local":
                fout.close()
            elif self.machine == "ccv" and not self.NN:
                with open(fns["lda_prep"], 'w') as f:
                    for element in fout:
                        f.write(str(element)+"\n")
            # BUG: I'm ignoring the case where self.NN AND self.machine == "ccv".
            # This is because currently we're not doing any preprocessing on the
            # neural network input. Should add another condition if we do at
            # some point

            if self.vote_counting and self.machine == "local":
                vote.close()
            elif self.vote_counting:
                with open(fns["votes"], 'w') as f:
                    for element in vote:
                        f.write(str(element)+"\n")

            if self.write_original and self.machine == "local":
                foriginal.close()
                main_indices.close()
            elif self.write_original and self.machine == "ccv":
                with open(fns["original_comm"], 'w') as f:
                    for element in foriginal:
                        f.write(str(element)+"\n")
                with open(fns["original_indices"], 'w') as f:
                    for element in main_indices:
                        f.write(str(element)+"\n")

            if self.author and self.machine == "local":
                author.close()
            elif self.author and self.machine == "ccv":
                with open(fns["author"],'w') as f:
                    for element in author:
                        f.write(str(element)+"\n")

            if self.sentiment and self.machine == "local":
                v_sentiments.close()
                t_sentiments.close()
                if not self.add_sentiment:
                    sentiments.close()

            elif self.sentiment and self.machine == "ccv":

                assert len(v_sentiments) == per_file_counter
                assert len(t_sentiments) == per_file_counter

                if not self.add_sentiment:
                    assert len(sentiments) == per_file_counter

                    with open(fns["sentiments"],'w') as f:
                        for element in sentiments:
                            f.write(str(element)+"\n")

                with open(fns["v_sentiments"], 'w') as g:
                    for element in v_sentiments:
                        g.write(str(element)+"\n")
                with open(fns["t_sentiments"], 'w') as h:
                    for element in t_sentiments:
                        h.write(str(element)+"\n")

            ccount.write(str(per_file_counter)+"\n")
            ccount.close()
            with open(fns["timedict"], "wb") as wfh:
                pickle.dump(timedict, wfh)

        # reset the missing files list for the next month
        missing_parsing_files = []

        # timer
        print("Finished parsing " + filename + " at  " + time.strftime('%l:%M%p, %m/%d/%Y'))

        # if the user wishes compressed data files to be removed after processing
        if self.clean_raw and filename not in self.on_file and Path(self.data_path + filename).is_file():
            print("Cleaning up {}{}.".format(self.data_path, filename))
            # delete the recently processed file
            os.system('cd {} && rm {}'.format(self.data_path, filename))

        return

    ## Pool parsed monthly data
    def pool_parsing_data(self):
        fns = self.get_parser_fns()
        # Initialize an "overall" timedict
        timedict = defaultdict(lambda: 0)
        for kind in fns.keys():
            fns_ = [self.get_parser_fns(year, month)[kind] for year, month in
                    self.dates]
            if kind == "counts":
                continue
            if kind == "timedict":
                # Update overall timedict with data from each year
                for fn_ in fns_:
                    with open(fn_, "rb") as rfh:
                        minitimedict = pickle.load(rfh)
                        for mo, val in minitimedict.items():
                            timedict[mo] += val
                with open(fns["timedict"], "w") as tdfh:
                    with open(fns["counts"], "w") as cfh:
                        cumul_docs = 0
                        for date in self.dates:
                            month = format_date(*date)
                            docs = timedict[month]
                            print(month + " " + str(docs), end='\n', file=tdfh)
                            # Use timedict data to populate counts file
                            cumul_docs += docs
                            print(cumul_docs, end='\n', file=cfh)
                continue
            subprocess.call("cat " + " ".join(fns_) + "> " + fns[kind], shell=True)

    ## Calls the multiprocessing module to parse raw data in parallel
    def parse(self,num_process=num_process):
        # get the correct hashsums to check file integrity
        #   self.hashsums = self.Get_Hashsums()

        # check for failed downloads and parse those months again
        if Path(self.model_path+"Download_Errors.txt").is_file():
            with open(self.model_path+"Download_Errors.txt","r") as file:
                for line in file:
                    year,month = int(line.strip().split(","))
                    if (year,month) not in self.dates:
                        self.dates.append((year,month))
                    if Path(self.data_path+ get_rc_filename(year,month)).is_file:
                        print("Corrupt download detected. Cleaning up {}{}.".format(self.data_path, filename))
                        os.system('cd {} && rm {}'.format(self.data_path, filename))
            os.system('cd {} && rm {}'.format(self.data_path, "Download_Errors.txt"))

        # Parallelize parsing by month
        inputs = [(year, month, self.on_file, self.__dict__) for year, month in self.dates]

        if self.machine == "ccv":
            try:
                current_batch = self.array * num_process
                previous_batch = max(0,self.array - 1 * num_process)

                if current_batch > len(inputs)-1 and previous_batch >= len(inputs)-1:
                    pass
                elif current_batch >= len(inputs)-1 and previous_batch < len(inputes)-1:

                    mpi_batch = inputs[current_batch:min(current_batch+num_process,len(inputs)-1)]
                    for input in mpi_batch:
                        self.parse_one_month(input[0],input[1])
                    self.pool_parsing_data()
                    self.lang_filtering() # filter non-English posts

                else:

                    mpi_batch = inputs[current_batch:min(current_batch+num_process,len(inputs)-1)]

                    # NOTE: For best results, set the number of processes in the following
                    # line based on (number of physical cores)*(hyper-threading multiplier)
                    # -1 (for synchronization overhead)
                    pool = multiprocessing.Pool(processes=num_process)

                    pool.map(parse_one_month_wrapper, mpi_batch)

                    mpi_batch = []

            except:
                raise Exception("Error in receiving batch IDs from the cluster.")

        elif self.machine == "local":

            pool = multiprocessing.Pool(processes=num_process)

            pool.map(parse_one_month_wrapper, inputs)

            # Pool parsing data from all files
            self.pool_parsing_data()
            self.lang_filtering() # filter non-English posts

        # timer
        print("Finished parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))

    ## Function to safely create folder structure for parsed files
    def safe_dir_create(self):
        fns = self.get_parser_fns()
        for key in fns:
            try:
                new_path = os.path.join(self.model_path, key)
                os.makedirs(new_path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(model_path):
                    continue
                else:
                    raise

    ## Function to call parser when needed and parse comments
    # TODO: Replace mentions of Vote in this file with mentions of sample_ratings
    # TODO: Add main counter and original comments and indices to this function
    def Parse_Rel_RC_Comments(self,num_process=num_process):
        # if preprocessed comments are available, ask if they should be rewritten
        if (self.NN and Path(self.model_path + "/bert_prep/bert_prep").is_file()) or (
                not self.NN and Path(self.model_path + "/lda_prep/lda_prep").is_file()):
            if self.machine == "local":
                Q = input(
                    "Preprocessed comments are already available. Do you wish to delete them and parse again [Y/N]?")
                if Q == 'Y' or Q == 'y':  # if the user wishes to overwrite the comments
                    # delete previous preprocessed data
                    if self.NN:  # for NN
                        shutil.rmtree(self.model_path + "/bert_prep")
                    elif not self.NN:  # for LDA
                        shutil.rmtree(self.model_path + "/lda_prep")
                    if Path(self.model_path + "/original_indices/original_indices").is_file() and self.write_original:
                        shutil.rmtree(self.model_path + "/original_indices")
                    if Path(self.model_path + "/original_comm/original_comm").is_file() and self.write_original:
                        shutil.rmtree(self.model_path + "/original_comm")
                    if Path(self.model_path + "/votes/votes").is_file() and self.vote_counting:
                        shutil.rmtree(self.model_path + "/votes")
                    if Path(self.model_path + "/author/author").is_file() and self.author:
                        shutil.rmtree(self.model_path + "/author")
                    if self.sentiment:
                        if Path(self.model_path + "/t_sentiments/t_sentiments").is_file():
                            shutil.rmtree(self.model_path + "/t_sentiments")
                        if Path(self.model_path + "/v_sentiments/v_sentiments").is_file():
                            shutil.rmtree(self.model_path + "/v_sentiments")
                        if not self.add_sentiment:
                            if Path(self.model_path + "/sentiments/sentiments").is_file():
                                shutil.rmtree(self.model_path + "/sentiments")
                    if Path(self.model_path + "counts/RC_Count_List").is_file():
                        shutil.rmtree(self.model_path + "/counts")
                    if Path(self.model_path + "timedict/RC_Count_Dict").is_file():
                        shutil.rmtree(self.model_path + "timedict")

                    # timer
                    print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
                    self.parse(num_process)

                else:  # if preprocessed comments are available and
                    # the user does not wish to overwrite them
                    print("Checking for missing files")

                    # check for other required files aside from main data
                    missing_files = 0

                    if not Path(self.model_path + "counts/RC_Count_List").is_file():
                        missing_files += 1

                    if not Path(self.model_path + "votes/votes").is_file() and self.vote_counting:
                        missing_files += 1

                    if not Path(self.model_path + "author/author").is_file() and self.author:
                        missing_files += 1

                    if self.sentiment:
                        if not Path(self.model_path + "v_sentiments/v_sentiments").is_file() or not Path(self.model_path + "t_sentiments/t_sentiments").is_file():
                            missing_files += 1
                        if not self.add_sentiment:
                            if not Path(self.model_path + "/sentiments/sentiments").is_file():
                                missing_files += 1

                    # if there are missing files, delete any partial record and parse again
                    if missing_files != 0:
                        print("Deleting partial record and parsing again")

                        if Path(self.model_path + "votes/votes").is_file():
                            shutil.rmtree(self.model_path + "votes")

                        if Path(self.model_path + "author/author").is_file():
                            shutil.rmtree(self.model_path + "author")

                        if Path(self.model_path + "t_sentiments/t_sentiments").is_file():
                            shutil.rmtree(self.model_path + "t_sentiments")

                        if Path(self.model_path + "v_sentiments/v_sentiments").is_file():
                            shutil.rmtree(self.model_path + "v_sentiments")

                        if Path(self.model_path + "/sentiments/sentiments").is_file():
                            shutil.rmtree(self.model_path + "/sentiments")

                        if self.NN:  # for NN
                            shutil.rmtree(self.model_path + "bert_prep")

                        elif not self.NN:  # for LDA
                            shutil.rmtree(self.model_path + "lda_prep")

                        if Path(self.model_path + "counts/RC_Count_List").is_file():
                            shutil.rmtree(self.model_path + "counts")

                        if Path(self.model_path + "timedict/RC_Count_Dict").is_file():
                            shutil.rmtree(self.model_path + "timedict")

                        # timer
                        print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
                        self.parse(num_process)

            elif self.machine == "ccv": # if running on the cluster

                print("Checking for missing files")

                # check for other required files aside from main data
                missing_files = 0

                if not Path(self.model_path + "counts/RC_Count_List").is_file():
                    missing_files += 1

                if not Path(self.model_path + "votes/votes").is_file() and self.vote_counting:
                    missing_files += 1

                if not Path(self.model_path + "author/author").is_file() and self.author:
                    missing_files += 1

                if self.sentiment:
                    if not Path(self.model_path + "v_sentiments/v_sentiments").is_file() or not Path(self.model_path + "t_sentiments/t_sentiments").is_file():
                        missing_files += 1
                    if not self.add_sentiment:
                        if not Path(self.model_path + "/sentiments/sentiments").is_file():
                            missing_files += 1

                # if there are missing files, delete any partial record and parse again
                if missing_files != 0:
                    print("Deleting partial record and parsing again")

                    if Path(self.model_path + "votes/votes").is_file():
                        shutil.rmtree(self.model_path + "votes")

                    if Path(self.model_path + "author/author").is_file():
                        shutil.rmtree(self.model_path + "author")

                    if Path(self.model_path + "sentiments/sentiments").is_file():
                        shutil.rmtree(self.model_path + "sentiments")

                    if Path(self.model_path + "t_sentiments/t_sentiments").is_file():
                        shutil.rmtree(self.model_path + "t_sentiments")

                    if Path(self.model_path + "v_sentiments/v_sentiments").is_file():
                        shutil.rmtree(self.model_path + "v_sentiments")

                    if self.NN:  # for NN
                        shutil.rmtree(self.model_path + "bert_prep")

                    elif not self.NN:  # for LDA
                        shutil.rmtree(self.model_path + "lda_prep")

                    if Path(self.model_path + "counts/RC_Count_List").is_file():
                        shutil.rmtree(self.model_path + "counts")

                    if Path(self.model_path + "timedict/RC_Count_Dict").is_file():
                        shutil.rmtree(self.model_path + "timedict")

                    # timer
                    print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
                    self.parse(num_process)

        else: # if aggregated dataset records do not exist, start parsing
            if Path(self.model_path + "counts/RC_Count_List").is_file():
                shutil.rmtree(self.model_path + "counts")
            if Path(self.model_path + "votes/votes").is_file() and self.vote_counting:
                shutil.rmtree(self.model_path + "votes")
            if Path(self.model_path + "author/author").is_file() and self.author:
                shutil.rmtree(self.model_path + "author")
            if self.sentiment:
                if Path(self.model_path + "t_sentiments/t_sentiments").is_file():
                    shutil.rmtree(self.model_path + "t_sentiments")
                if Path(self.model_path + "v_sentiments/v_sentiments").is_file():
                    shutil.rmtree(self.model_path + "v_sentiments")
                if not self.add_sentiment:
                    if Path(self.model_path + "/sentiments/sentiments").is_file():
                        shutil.rmtree(self.model_path + "/sentiments")
            if Path(self.model_path + "original_comm/original_comm").is_file() and self.write_original:
                shutil.rmtree(self.model_path + "original_comm")
            if Path(self.model_path + "original_indices/original_indices").is_file() and self.write_original:
                shutil.rmtree(self.model_path + "original_indices")

            # timer
            print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
            self.parse(num_process)

    ## Function for removing non-English posts picked up by the regex filter
    def lang_filtering(self):

        if Path(self.model_path + "/non_en").is_file():  # if corpus is already filtered
            print("Found language filtering results on file. Moving on.")
            pass
        else:  # otherwise

            # check for missing files per parameter configs

            # raw dataset
            if not Path(self.model_path + "/original_comm/original_comm").is_file():
                raise Exception('Original comments are needed and could not be found')
            if not Path(self.model_path + "/original_indices/original_indices").is_file():
                raise Exception('Original indices are needed and could not be found')

            # preprocessed data
            if (not Path(self.model_path + "/lda_prep/lda_prep").is_file()) and self.NN == False:
                raise Exception('Preprocessed dataset could not be found')
            elif (not Path(self.model_path + "/bert_prep/bert_prep.json").is_file()) and self.NN == True:
                raise Exception('Preprocessed dataset could not be found')

            # cumulative post counts
            if not Path(self.model_path + "/counts/RC_Count_List").is_file():
                raise Exception(
                    'Cumulative monthly comment counts could not be found')
            else:  # load the cumulative counts
                timelist_original = []
                with open(self.model_path + "/counts/RC_Count_List", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            timelist_original.append(int(float(line.strip())))

            # get the file counts
            file_counts = []
            for element in self.dates:
                yr,mo = element[0],element[1]
                with open(self.model_path + "/counts/RC_Count_List-{}-{}".format(yr,mo),"r") as file_count:
                    for line in file_count:
                        if line.strip() != "":
                            file_counts.append(int(line.strip()))
            assert len(file_counts) == len(self.dates)

            # post meta-data
            if (not Path(self.model_path + "/votes/votes").is_file()) and self.vote_counting:
                raise Exception('Votes counld not be found')
            if (not Path(self.model_path + "/author/author").is_file()) and self.author:
                raise Exception('Author usernames could not be found')
            if self.sentiment:
                if not Path(self.model_path + "/v_sentiments/v_sentiments").is_file():
                    raise Exception('Vader sentiment estimates could not be found')
                if not Path(self.model_path + "/t_sentiments/t_sentiments").is_file():
                    raise Exception('TextBlob sentiment estimates could not be found')
                if not self.add_sentiment:
                    if not Path(self.model_path + "/sentiments/sentiments").is_file():
                        raise Exception('Sentiment estimates could not be found')

            fns = self.get_parser_fns()

            # Initialize variables

            # timer
            print("Started filtering out non-English posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))

            # seed the random initializer
            DetectorFactory.seed = 0

            # counters for the number of non-English posts from each time period
            int_non_en = np.zeros(len(timelist_original)+1)
            file_non_en = np.zeros(len(timelist_original)+1)

            non_en_idx = []  # list for indices of non-English posts

            int_counter = 0  # counter for the time period an index belongs to
            file_counter = 0

            # Filter the posts

            with open(self.model_path + "/non_en", "w") as non_en:
                total_count = 0
                with open(self.model_path + "/original_comm/original_comm", "r") as raw_dataset:
                    for index, post in enumerate(raw_dataset):
                        try:
                            if index == timelist_original[int_counter]:
                                int_counter += 1  # update time interval counter
                        except:
                            int_counter += 1
                        try:
                            if index == sum(file_counts[:file_counter+1]):
                                file_counter += 1
                        except:
                            file_counter += 1
                        try:  # if post is too short to reliably analyze or
                            # highly likely to be in English
                            if detect(post) == 'en' or len(post.split()) <= 20:
                                pass
                            else:  # if post is likely not to be in English
                                non_en_idx.append(index)  # record the index
                                int_non_en[int_counter] += 1  # update
                                file_non_en[file_counter] += 1
                                # non-English post counter
                                print(index, end="\n", file=non_en)  # save index
                        except:  # if language identification failed, add the
                            # post to the list of posts to be removed from dataset
                            non_en_idx.append(index)  # record the index
                            int_non_en[int_counter] += 1  # update non-English
                            file_non_en[file_counter] += 1
                            # post counter
                            print(index, end="\n", file=non_en)  # save index
                        total_count += 1

            # A list of dataset files needing to be updated based on parameters
            filenames = ['/original_comm/original_comm','/original_indices/original_indices']
            if not self.NN:
                filenames.append("/lda_prep/lda_prep")
            if self.vote_counting:
                filenames.append("/votes/votes")
            if self.author:
                filenames.append("/author/author")
            if self.sentiment:
                filenames.append("/v_sentiments/v_sentiments")
                filenames.append("/t_sentiments/t_sentiments")
                if not self.add_sentiment:
                    filenames.append("/sentiments/sentiments")

            for file in filenames: # for each file in the list above

                with open(self.model_path + file,"r") as f: # read each line
                    lines = f.readlines()
                with open(self.model_path + file, "w") as f: # write only the relevant posts
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

                # BUG: this hacky solution would only work for fully consecutive
                # months within self.dates --> make it more general
                # update monthly files
                total_counter = 0
                for yr,mo in self.dates:
                    with open(self.model_path + file +"-{}-{}".format(yr,mo),"r") as monthly_file:
                        lines = monthly_file.readlines()
                    with open(self.model_path + file +"-{}-{}".format(yr,mo),"w") as monthly_file:
                        for line in lines:
                            if line.strip() != "":
                                if total_counter not in non_en_idx:
                                    monthly_file.write(line)
                                total_counter += 1

            # update document counts for each time interval in the dataset
            running_tot_count = 0
            for interval, count in enumerate(timelist_original):
                running_tot_count += int_non_en[interval]
                timelist_original[interval] = timelist_original[interval] - running_tot_count

            #BUG: bert_prep is not being updated. Okay for now, but bad if we'll be using it

            with open(self.model_path + "/counts/RC_Count_List", "w") as f:
                for interval in timelist_original:
                    print(int(interval), end="\n", file=f)

            for idx,element in enumerate(self.dates):
                yr,mo = element[0],element[1]
                with open(self.model_path+"/counts/RC_Count_List-{}-{}".format(yr,mo),"w") as f:
                    new_count = file_counts[idx] - file_non_en[idx]
                    f.write(str(int(new_count)))

            # timer
            print("Finished filtering out non-English posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))

    # TODO: add information
    def Screen_One_Month(self, year, month):

        # set up neural network runtime configurations
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        # load the pre-trained neural network from model_path
        model = ClassificationModel('roberta', rel_model_path, use_cuda=False,
                                    args={'fp16': False, 'num_train_epochs': 1, 'manual_seed': 1, 'silent': True})

        total_count = 0  # counter for all documents in the dataset

        start = time.time()  # measure processing time

        # if labels for that month exist, load them
        if Path(self.model_path + "/auto_labels/auto_labels-{}-{}".format(year, month)).is_file():
            print("Found labels for year " + str(year) + ", month " + str(month) + ". Loading.")

            with open(self.model_path + "/auto_labels/auto_labels-{}-{}".format(year, month), "r") as labels:
                for label in labels:
                    if label.strip() != "":
                        total_count += 1

        else:  # otherwise use the trained network to obtain them

            # get the number of documents within each parsed monthly data file
            # NOTE: Might be slightly different from the actual
            # monthly counts
            file_counter = 0
            with open(self.model_path + "/counts/RC_Count_List-{}-{}".format(year, month), "r") as counts:
                for line in counts:
                    if line.strip() != "":
                        file_counter = int(line)
                        total_count += int(line)

            # use original comment text to auto-generate labels

            # create folder for automatic relevance labels if none exists
            if not os.path.isdir(self.model_path + "/auto_labels"):
                os.system('cd {} && mkdir {}'.format(self.data_path, "auto_labels"))

            with open(self.model_path + "/original_comm/original_comm-{}-{}".format(year, month), "r") as texts, open(
                    self.model_path + "/auto_labels/auto_labels-{}-{}".format(year, month), "w") as labels:

                counter = 0  # doc counter for batching

                batch = []
                for line in texts:

                    batch.append(line)
                    counter += 1

                    if counter % 100 == 0 or counter == file_counter:

                        prediction, raw_output = model.predict(batch)
                        for element in prediction:
                            labels.write(str(element) + "\n")

                        batch = []

        # calculate and report processing time
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Processed month {} of year {} in {:0>2}:{:0>2}:{:05.2f}".format(month, year, int(hours), int(minutes),
                                                                               seconds))

    ## Uses a pre-trained neural network to prune dataset from irrelevant posts
    def Neural_Relevance_Screen(self, rel_model_path=rel_model_path, dates=dates,
                                rel_sample_num=rel_sample_num, balanced_rel_sample=balanced_rel_sample):

        # BUG: Add pooling function to the end
        total_count = 0

        # check for previous screening results
        if Path(self.model_path + "/auto_labels/sample_auto_labeled.csv").is_file():

            print(
                "A sample of auto-labeled posts was found, suggesting neural relevance screening was previously performed. Moving on.")

        else:  # if screening results not found

            # check for a complete record of labels
            if Path(self.model_path + "/auto_labels/auto_labels").is_file():
                print("Found full dataset labels. Loading.")

                with open(self.model_path + "/auto_labels/auto_labels", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            total_count += 1

            else:  # if no previous record is found

                # TODO: Adjust the following to only get the months in dates.
                # Same for other parsing functions

                # Load cumulative number of relevant posts for each month, from disk

                if not Path(self.model_path + "/counts/RC_Count_List").is_file():
                    raise Exception(
                        'Cumulative monthly comment counts could not be found')
                else:
                    timelist_original = []
                    with open(self.model_path + "/counts/RC_Count_List", "r") as f:
                        for line in f:
                            if line.strip() != "":
                                timelist_original.append(int(line))
                                total_count += int(line)

            # Parallelize parsing by month

            inputs = [(year, month, self.on_file, self.__dict__) for year, month in self.dates]

            if self.machine == "ccv":

                try:
                    current_batch = self.array
                    if current_batch > len(inputs)-1:
                        pass
                    else:
                        self.Screen_One_Month(current_batch[0],current_batch[1])

                        if current_batch == len(inputs)-1:

                            with open(self.model_path + "/auto_labels/auto_labels", "a+") as general_labels:
                                for yr, mo,_,_ in inputs:
                                    with open(self.model_path + "/auto_labels/auto_labels-{}-{}".format(yr, mo),
                                              "r") as monthly:
                                        for line in monthly:
                                            if line.strip() != "":
                                                general_labels.write(line.strip() + "\n")

                            # Sample rel_sample_num documents at random for evaluating the dataset and the auto-labeling
                            random_sample = list(
                                np.random.choice(range(0, total_count), size=rel_sample_num, replace=False))
                            labels_array = np.ones(total_count)  # initiate array for labels

                            # read labels from disk and identify indices of irrelevant posts
                            irrel_idxes = []
                            with open(self.model_path + "/auto_labels/auto_labels", "r") as labels:
                                for idx, line in enumerate(labels):
                                    if line.strip() == '0' or line.strip() == 'None':
                                        labels_array[idx] = 0
                                        irrel_idxes.append(idx)

                            # TODO: Add the ability to deal with posts for which the classifier
                            # has failed to provide a prediction but rather an error (None classes).

                            # if obtaining a sample of labels balanced across output categories:
                            if balanced_rel_sample:
                                sampled_cats = {0: 0, 1: 0}
                                goal = {0: floor(float(rel_sample_num) / 2), 1: ceil(float(rel_sample_num) / 2)}

                                for value in random_sample:
                                    sampled_cats[int(labels_array[value])] += 1

                                print("Random sample composition: ")
                                print(sampled_cats)

                                for category, _ in sampled_cats.items():  # upsample and downsample
                                    # to have a balance of posts marked positive or negative

                                    while sampled_cats[category] < goal[category]:

                                        if category == 0:
                                            relevant_subset = [i for i in irrel_idxes if i not in random_sample]
                                        elif category == 1:
                                            relevant_subset = [i for i in range(0, total_count) if
                                                               i not in irrel_idxes and i not in random_sample]

                                        new_proposed = np.random.choice(relevant_subset)

                                        random_sample.append(new_proposed)
                                        sampled_cats[category] += 1
                                        relevant_subset.remove(new_proposed)

                                    while sampled_cats[category] > goal[category]:

                                        if category == 0:
                                            relevant_subset = [i for i in random_sample if i in irrel_idxes]
                                        elif category == 1:
                                            relevant_subset = [i for i in random_sample if i not in irrel_idxes]

                                        new_proposed = np.random.choice(relevant_subset)
                                        random_sample.remove(new_proposed)
                                        sampled_cats[category] -= 1

                                print("Balanced random sample counts: ")
                                print(sampled_cats)

                            # Sample the documents using the chosen indices
                            print("Sampling " + str(len(set(random_sample))) + " documents")

                            sampled_docs = []
                            int_counter = 0

                            with open(self.model_path + "/original_comm/original_comm", "r") as sampling:
                                sampled_idxes = []
                                for idx, sampler in enumerate(sampling):
                                    try:
                                        if idx > timelist_original[int_counter]:
                                            int_counter += 1
                                        if idx in random_sample:
                                            # writing year, month, text to sample file
                                            sampled_docs.append(
                                                [str(dates[int_counter][0]), str(dates[int_counter][1]), sampler])
                                    except:  # if the date associated with the post is beyond
                                        # the pre-specified time intervals, associate the post with
                                        # the next month. May happen with a few documents at the edges
                                        # of data files

                                        if idx in random_sample:

                                            # determine the relevant next month and year
                                            yr = dates[int_counter - 1][0]
                                            mo = dates[int_counter - 1][1]
                                            if mo == 12:
                                                yr += 1
                                                mo = 1
                                            else:
                                                mo += 1

                                            sampled_docs.append([str(yr), str(mo), sampler])

                            # load labels for the sampled documents
                            general_counter = 0
                            sample_counter = 0
                            with open(self.model_path + "/auto_labels/auto_labels", "r") as labels:
                                for idx, line in enumerate(labels):
                                    if idx in random_sample:
                                        sampled_docs[sample_counter].append(line)
                                        sample_counter += 1

                            # shuffle the docs and check number and length
                            np.random.shuffle(sampled_docs)
                            try:
                                assert len(sampled_docs) == rel_sample_num
                            except:
                                assert len(sampled_docs) == len(irrel_idxes)
                            for element in sampled_docs:
                                assert len(element) == 4

                            # write the sampled files to a csvfile
                            with open(self.model_path + "/auto_labels/sample_auto_labeled.csv", 'a+') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(['year', 'month', 'text', 'auto label', 'accuracy'])
                                for document in sampled_docs:
                                    writer.writerow(document)

                        # break

                except:
                    raise Exception("Error in receiving batch IDs from the cluster.")

            elif self.machine == "local":

                for yr, month, _, _ in inputs:
                    self.Screen_One_Month(yr, month)

                # Sample rel_sample_num documents at random for evaluating the dataset and the auto-labeling
                random_sample = list(np.random.choice(range(0, total_count), size=rel_sample_num, replace=False))
                labels_array = np.ones(total_count)  # initiate array for labels

                with open(self.model_path + "/auto_labels/auto_labels", "a+") as general_labels:
                    for yr, mo, _, _ in inputs:
                        with open(self.model_path + "/auto_labels/auto_labels-{}-{}".format(yr, mo), "r") as monthly:
                            for line in monthly:
                                if line.strip() != "":
                                    general_labels.write(line.strip() + "\n")

                # read labels from disk and identify indices of irrelevant posts
                irrel_idxes = []
                with open(self.model_path + "/auto_labels/auto_labels", "r") as labels:
                    for idx, line in enumerate(labels):
                        if line.strip() == '0' or line.strip() == 'None':
                            labels_array[idx] = 0
                            irrel_idxes.append(idx)

                # TODO: Add the ability to deal with posts for which the classifier
                # has failed to provide a prediction but rather an error (None classes).

                # if obtaining a sample of labels balanced across output categories:
                if balanced_rel_sample:
                    sampled_cats = {0: 0, 1: 0}
                    goal = {0: floor(float(rel_sample_num) / 2), 1: ceil(float(rel_sample_num) / 2)}

                    for value in random_sample:
                        sampled_cats[int(labels_array[value])] += 1

                    print("Random sample composition: ")
                    print(sampled_cats)

                    for category, _ in sampled_cats.items():  # upsample and downsample
                        # to have a balance of posts marked positive or negative

                        while sampled_cats[category] < goal[category]:

                            if category == 0:
                                relevant_subset = [i for i in irrel_idxes if i not in random_sample]
                            elif category == 1:
                                relevant_subset = [i for i in range(0, total_count) if
                                                   i not in irrel_idxes and i not in random_sample]

                            new_proposed = np.random.choice(relevant_subset)

                            random_sample.append(new_proposed)
                            sampled_cats[category] += 1
                            relevant_subset.remove(new_proposed)

                        while sampled_cats[category] > goal[category]:

                            if category == 0:
                                relevant_subset = [i for i in random_sample if i in irrel_idxes]
                            elif category == 1:
                                relevant_subset = [i for i in random_sample if i not in irrel_idxes]

                            new_proposed = np.random.choice(relevant_subset)
                            random_sample.remove(new_proposed)
                            sampled_cats[category] -= 1

                    print("Balanced random sample counts: ")
                    print(sampled_cats)

                # Sample the documents using the chosen indices
                print("Sampling " + str(len(set(random_sample))) + " documents")

                sampled_docs = []
                int_counter = 0

                with open(self.model_path + "/original_comm/original_comm", "r") as sampling:
                    sampled_idxes = []
                    for idx, sampler in enumerate(sampling):
                        try:
                            if idx > timelist_original[int_counter]:
                                int_counter += 1
                            if idx in random_sample:
                                # writing year, month, text to sample file
                                sampled_docs.append([str(dates[int_counter][0]), str(dates[int_counter][1]), sampler])
                        except:  # if the date associated with the post is beyond
                            # the pre-specified time intervals, associate the post with
                            # the next month. May happen with a few documents at the edges
                            # of data files

                            if idx in random_sample:

                                # determine the relevant next month and year
                                yr = dates[int_counter - 1][0]
                                mo = dates[int_counter - 1][1]
                                if mo == 12:
                                    yr += 1
                                    mo = 1
                                else:
                                    mo += 1

                                sampled_docs.append([str(yr), str(mo), sampler])

                # load labels for the sampled documents
                general_counter = 0
                sample_counter = 0
                with open(self.model_path + "/auto_labels/auto_labels", "r") as labels:
                    for idx, line in enumerate(labels):
                        if idx in random_sample:
                            sampled_docs[sample_counter].append(line)
                            sample_counter += 1

                # shuffle the docs and check number and length
                np.random.shuffle(sampled_docs)
                assert len(sampled_docs) == rel_sample_num
                for element in sampled_docs:
                    assert len(element) == 4

                # write the sampled files to a csvfile
                with open(self.model_path + "/auto_labels/sample_auto_labeled.csv", 'a+') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['year', 'month', 'text', 'auto label', 'accuracy'])
                    for document in sampled_docs:
                        writer.writerow(document)

    ## Function that uses the results of Neural_Relevance_Screen to remove posts
    # likely to be irrelevant from the dataset.
    def Neural_Relevance_Clean(self, model_path=model_path, dates=dates):

        # check to see if relevance filtering was previously performed
        if Path(self.model_path + "/rel_clean_cert.txt").is_file():
            print("A certificate for a previously finished relevance filtering was found. Moving on.")

        else:  # if not

            # check for needed but missing files per parameter configs
            # BUG: bert_prep is not being updated. Okay for now, but will be problematic
            # if we end up using it

            # raw dataset and indices
            if not Path(self.model_path + "/original_comm/original_comm").is_file():
                raise Exception('Original comments could not be found')
            if not Path(self.model_path + "/original_indices/original_indices").is_file():
                raise Exception('Original indices could not be found')

            # preprocessed data for LDA and NN
            if (not Path(self.model_path + "/lda_prep/lda_prep").is_file()) and self.NN == False:
                raise Exception('Preprocessed dataset could not be found')
            elif (not Path(self.model_path + "/bert_prep/bert_prep.json").is_file()) and self.NN == True:
                raise Exception('Preprocessed dataset could not be found')

            # cumulative post counts
            if not Path(self.model_path + "/counts/RC_Count_List").is_file():
                raise Exception(
                    'Cumulative monthly comment counts could not be found')
            else:  # load the cumulative counts
                timelist_original = []
                with open(self.model_path + "/counts/RC_Count_List", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            timelist_original.append(int(line))

            # post meta-data
            if (not Path(self.model_path + "/votes/votes").is_file()) and self.vote_counting:
                raise Exception('Votes counld not be found')
            if (not Path(self.model_path + "/author/author").is_file()) and self.author:
                raise Exception('Author usernames could not be found')
            if self.sentiment:
                if not Path(self.model_path + "/v_sentiments/v_sentiments").is_file():
                    raise Exception('Vader sentiment estimates could not be found')
                if not Path(self.model_path + "/t_sentiments/t_sentiments").is_file():
                    raise Exception('TextBlob sentiment estimates could not be found')
                if not self.add_sentiment:
                    if not Path(self.model_path + "/sentiments/sentiments").is_file():
                        raise Exception('Sentiment estimates could not be found')

            # Initialize variables

            # timer
            print("Started filtering out irrelevant posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))
            start = time.time()  # start a clock for processing time

            # load negative or unknown labels for the dataset from disk

            int_counter = 0  # counter for the time period an index belongs to

            # counters for the number of irrelevant posts from each time period
            int_non_rel = np.zeros(len(timelist_original) + 1)

            # counters for the number of irrelevant posts from each file
            file_non_rel = np.zeros_like(timelist_original)

            general_counter = 0  # counter for all comments
            file_counter = 0  # counter for comments in a monthly file
            negative_labels = []  # list for negative labels

            for yr, mo in dates:  # for each month
                with open(model_path + "/auto_labels/auto_labels-{}-{}".format(yr, mo), "r") as labels:
                    for idx, line in enumerate(labels):
                        if line.strip() != "":

                            try:
                                if general_counter > timelist_original[int_counter]:
                                    int_counter += 1  # try to update month counter as needed
                                # if label was 0 or there was an error in labeling, remove
                                if line.strip() == "None" or line.strip() == "0":
                                    negative_labels.append(general_counter)  # add index
                                    int_non_rel[int_counter] += 1  # update counter
                                    file_non_rel[file_counter] += 1  # update counter

                            except:  # if the date of the post is out of bounds for the file,
                                # ignored the requirement to update the monthly counter
                                # if label was 0 or there was an error in labeling, remove
                                if line.strip() == "None" or line.strip() == "0":
                                    negative_labels.append(general_counter)  # add index
                                    int_non_rel[int_counter + 1] += 1  # update counter
                                    file_non_rel[file_counter] += 1  # update counter

                            general_counter += 1  # update

                file_counter += 1  # update

            # Filter the posts

            # BUG: I'm not filtering bert_prep. It's okay as long as we don't use it

            # A list of dataset files needing to be updated based on parameters
            filenames = ['/original_comm/original_comm', '/original_indices/original_indices',
                         '/auto_labels/auto_labels']
            if not self.NN:
                filenames.append("/lda_prep/lda_prep")
            if self.vote_counting:
                filenames.append("/votes/votes")
            if self.author:
                filenames.append("/author/author")
            if self.sentiment:
                filenames.append("/v_sentiments/v_sentiments")
                filenames.append("/t_sentiments/t_sentiments")
                if not self.add_sentiment:
                    filenames.append("/sentiments/sentiments")

            for file in filenames:  # for each file in the list above

                with open(self.model_path + file, "r") as f:  # read each line
                    lines = f.readlines()
                with open(self.model_path + file, "w") as f:  # write only the relevant posts
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in negative_labels:
                            f.write(line)

                for yr, mo in self.dates:
                    with open(self.model_path + file + "-{}-{}".format(yr, mo), "r") as monthly_file:
                        lines = f.readlines()
                    with open(self.model_path + file + "-{}-{}".format(yr, mo), "w") as monthly_file:
                        for index, line in enumerate(monthly_file):
                            if line.strip() != "" and index not in negative_labels:
                                monthly_file.write(line)

            # update document counts for each time interval in the dataset
            running_tot_count = 0
            for interval, count in enumerate(timelist_original):
                running_tot_count += int_non_rel[interval]
                timelist_original[interval] = timelist_original[interval] - running_tot_count

            # get the actual post count (might be different from the list above
            # because of out-of-bounds posts, for which an additional entry will
            # be appended to the count list)
            with open(model_path + '/original_comm/original_comm', 'r') as f:
                total_count = 0
                for line in f:
                    total_count += 1
                if not timelist_original[-1] == total_count:
                    timelist_original.append(total_count)

            # update the cumulative monthly counts
            with open(self.model_path + "/counts/RC_Count_List", "w") as f:
                for interval in timelist_original:
                    print(str(int(interval)), end="\n", file=f)

            # fix monthly file counts
            total_counter = 0
            for yr, mo in dates:
                file_counter = 0
                interval_counter = 0
                with open(self.model_path + "/counts/RC_Count_List-{}-{}".format(yr, mo), "r") as f:
                    for line in f:
                        if line.strip() != "":
                            file_counter = int(line.strip())

                for index in negative_labels:
                    if index >= total_counter and index < total_counter + file_counter:
                        interval_counter += 1
                total_counter += int(line.strip())
                with open(self.model_path + "/counts/RC_Count_List-{}-{}".format(yr, mo), "w") as f:
                    new_count = file_counter - interval_counter
                    f.write(str(new_count))

            # update the auto-labels file and check that all negative comments
            # are removed from the dataset
            count_auto_labels = 0
            sum_auto_labels = 0
            with open(self.model_path + "/auto_labels/auto_labels", "r") as labels:
                for line in labels:
                    if line.strip() != "":
                        count_auto_labels += 1
                        sum_auto_labels += int(line.strip())
            assert count_auto_labels == sum_auto_labels

            # Measure, report and record filtering time for the entire dataset
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            with open(self.model_path+"rel_clean_cert.txt","w") as cert:
                cert.write("Finished relevance-filtering in {:0>2}:{:0>2}:{:05.2f} (hours,minutes,seconds)".format(int(hours),int(minutes),seconds))
                print("Finished relevance-filtering in {:0>2}:{:0>2}:{:05.2f} (hours,minutes,seconds)".format(int(hours),int(minutes),seconds))

    ## Evaluates accuracy, f1, precision and recall for the relevance classifier
    # based on the random sample from Neural_Relevance_Screen
    def eval_relevance(self, sample_path=model_path + "/auto_labels/"):

        # check that the random sample is available
        if not Path(sample_path + "sample_auto_labeled.csv").is_file:
            raise Exception("Random sample of classifier output not found.")

        else:  # if it is
            with open(sample_path + "sample_auto_labeled.csv", "r") as csvfile:
                reader = csv.reader(csvfile)

                labels = []  # container for human labels
                preds = []  # container for model predictions
                for idx, row in enumerate(reader):
                    if idx != 0:
                        if row[4].strip() == "":  # check for human labels
                            raise Exception("Random sample is not fully hand-annotated")
                        else:
                            preds.append(int(row[3].strip()))
                            labels.append(int(row[4].strip()))

        label_measures = dict()  # dictionary for confusion matrix

        accuracy = 0  # counter for accuracy

        # populate the confusion matrix
        for index, ind_label in enumerate(labels):
            if labels[index] == 1 and preds[index] == 1:
                if 'tp' in label_measures:
                    label_measures['tp'] += 1
                else:
                    label_measures['tp'] = 1
                accuracy += 1
            elif labels[index] == 1:
                if 'fn' in label_measures:
                    label_measures['fn'] += 1
                else:
                    label_measures['fn'] = 1
            elif labels[index] == 0 and preds[index] == 0:
                if 'tn' in label_measures:
                    label_measures['tn'] += 1
                else:
                    label_measures['tn'] = 1
                accuracy += 1
            elif labels[index] == 0:
                if 'fp' in label_measures:
                    label_measures['fp'] += 1
                else:
                    label_measures['fp'] = 1

        # print out the confusion matrix
        print("Confusion matrix: ")
        print(label_measures)

        # Check that values are assigned to measures needed for f1, precision
        # and recall
        try:
            tp = label_measures['tp']
        except:
            tp = 0
        try:
            fp = label_measures['fp']
        except:
            fp = 0
        try:
            fn = label_measures['fn']
        except:
            fn = 0

        # Write evaluation measures if they are calculable to file
        with open(sample_path + "eval_results.txt", "a+") as f:

            # Record the confusion matrix
            print("Confusion matrix: " + str(label_measures) + "\n")

            # Record the accuracy
            accuracy = float(accuracy) / float(len(labels))
            print("accuracy: " + str(accuracy))
            f.write("accuracy: " + str(accuracy) + "\n")

            if tp != 0 or fp != 0:  # record the precision
                precision = float(tp) / (float(tp) + float(fp))
                print("precision: " + str(precision))
                f.write("precision: " + str(precision) + "\n")

            if tp != 0 or fn != 0:  # record the recall
                recall = float(tp) / (float(tp) + float(fn))
                print("recall: " + str(recall))
                f.write("recalls: " + str(recall) + "\n")

            if tp != 0 or (fn != 0 and fp != 0):  # record the f1 score
                f1 = 2 * (precision * recall) / (precision + recall)
                print("f1: " + str(f1))
                f.write("f1s: " + str(f1) + "\n")

    ### Records CoreNLP sentiment estimates, parallelized via threading, and
    # writes to disk the average standardized sentiment of each post in the dataset
    # acording to Vader, TextBlob and CoreNLP

    # NOTE: Parsing should be finished for the months in dates before running
    # this function

    def add_sentiment(self):

        if not os.path.exists(self.model_path + "/c_sentiments/"):
            print("Creating directories to store the additional sentiment output")
            os.makedirs(self.model_path + "/c_sentiments")
        if not os.path.exists(self.model_path + "/sentiments/"):
            os.makedirs(self.model_path + "/sentiments")

        fns["c_sentiments"] = "{}/c_sentiments/c_sentiments".format(self.model_path)
        fns["sentiments"] = "{}/sentiments/sentiments".format(self.model_path)

        ## Check for needed files
        if not Path(self.model_path + "/original_comm/original_comm").is_file():
            raise Exception('Original comments are needed and could not be found')
        if not Path(self.model_path + "/v_sentiments/v_sentiments").is_file():
            raise Exception('Vader sentiment estimates could not be found')
        if not Path(self.model_path + "/t_sentiments/t_sentiments").is_file():
            raise Exception('TextBlob sentiment estimates could not be found')
        # cumulative post counts
        if not Path(self.model_path + "/counts/RC_Count_List").is_file():
            raise Exception(
                'Cumulative monthly comment counts could not be found')
        else:  # load the cumulative counts
            timelist_original = []
            with open(self.model_path + "/counts/RC_Count_List", "r") as f:
                for line in f:
                    if line.strip() != "":
                        timelist_original.append(int(float(line.strip())))

        # Create the containers based on whether running locally or on the cluster
        if self.machine == "ccv":
            c_sentiments = []
            sentiments = []
        elif self.machine == "local":
            c_sentiments = open(fns["c_sentiments"], "w")
            sentiments = open(fns["sentiments"],"w")

        ## Record CoreNLP estimates, while accounting for its numerous errors
        total_core_nlp = []
        count_core_nlp = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        with open(fns["original_comm"], 'r') as original_texts:

            for index,original_body in enumerate(original_texts):

                tokenized = sent_detector.tokenize(original_body)
                count_core_nlp.append(len(tokenized))

                original_without_quotes = original_body.replace("\"", "")

                per_sentence = []
                error_indicator = 0

                try:
                    annot_doc = self.nlp_wrapper.annotate(original_without_quotes,
                        properties={'annotators': 'sentiment','outputFormat': 'json',
                        'parse.maxlen':100,'parse.nthreads':self.num_cores-3,
                        'timeout': 100000 })
                except:
                    print("CoreNLP error")
                    error_indicator = 1

                    with open("CoreNLP_errors.txt", "a+") as errors:
                        errors.write(
                            str(index) + "," + " ".join(original_body.split()).replace("\n", ""))
                    total_core_nlp.append("None")
                    if self.machine == "local":
                        with open(fns["c_sentiments"], "a+") as c_sentiments:
                            c_sentiments.write("None" + "\n")
                    elif self.machine == "ccv":
                        c_sentiments.append("None")

                if error_indicator == 0:
                    try:
                        sum_core_nlp = 0
                        for i in range(0, len(annot_doc['sentences'])):
                            sum_core_nlp += int(annot_doc['sentences'][i]['sentimentValue'])
                            per_sentence.append(str(annot_doc['sentences'][i]['sentimentValue']))
                        total_core_nlp.append(sum_core_nlp)
                    except:
                        total_core_nlp.append("None")
                        per_sentence.append("None")

                    if len(per_sentence) != 0 and "None" not in per_sentence:
                        if self.machine == "local":
                            with open(fns["c_sentiments"], "a+") as c_sentiments:
                                c_sentiments.write(",".join(per_sentence) + "\n")
                        elif self.machine == "ccv":
                            c_sentiment.append(",".join(per_sentence))
                    else:
                        if self.machine == "local":
                            with open(fns["c_sentiments"], "a+") as c_sentiments:
                                c_sentiments.write("None")
                        elif self.machine == "ccv":
                            c_sentiments.append("None")

            # write per-sentence CoreNLP sentiments to disk
            if self.machine == "local":
                c_sentiments.close()
            elif self.machine == "ccv":
                with open(fns["c_sentiments"], "w") as c_sentiment:
                    for sentiment in c_sentiments:
                        c_sentiment.write(sentiment)

            ## read sentiment values for each post from the two other packages
            total_vader = []
            count_vader = []
            with open(fns["v_sentiments"], "r") as v_sentiments:
                for line in v_sentiments:
                    values = [float(i) for i in line.strip(",").split(",")]
                    count_vader.append(len(values))
                    total_vader.append(int(np.sum(values)))
            total_textblob = []
            count_textblob = []
            with open(fns["t_sentiments"], "r") as t_sentiments:
                for line in t_sentiments:
                    values = [float(i) for i in line.strip(",").split(",")]
                    count_textblob.append(len(values))
                    total_textblob.append(int(np.sum(values)))

            # Make sure all posts are accounted for in the three sets
            assert len(total_textblob) == len(total_vader)
            assert len(total_vader) == len(total_core_nlp)
            assert len(count_textblob) == len(count_vader)
            assert len(count_vader) == len(count_core_nlp)

            ## Calculate the average sentiment based on 2 or 3 packages, depending
            # on whether CoreNLP encountered an error
            for index,_ in enumerate(total_textblob):
                avg_vader = total_vader[index] / count_vader[index]
                avg_blob = total_textblob[index] / count_textblob[index]

                if total_core_nlp[index] != "None":
                    avg_core_nlp = total_core_nlp[index] / count_core_nlp[index]
                    # Normalizing core nlp so it's between -1 and 1
                    normalized_core_nlp = ((avg_core_nlp / 4) * 2) - 1
                    avg_score = (avg_vader + avg_blob + normalized_core_nlp) / 3
                else:
                    avg_score = (avg_vader + avg_blob) / 2

                if self.machine == "local":
                    print(avg_score, file=sentiments)
                elif self.machine == "ccv":
                    sentiments.append(avg_score)

            # Clean up and write to file
            if self.machine == "local":
                sentiments.close()
                c_sentiments.close()
            elif self.machine == "ccv":
                with open(fns["sentiments"], "w") as avg_sentiment:
                    for sentiment in sentiments:
                        avg_sentiment.write(sentiment)

            ## Create monthly files based on the aggregated one for CoreNLP and
            # and average sentiment estimates
            # BUG: hacky solution. Only works for consecutive sets of months
            int_counter = 0
            yr,mo = dates[int_counter]
            month_path = self.model_path + "/c_sentiments/c_sentiments-{}-{}".format(yr,mo)

            if not Path(month_path).is_file():
                if self.machine == "local":
                    sent_monthly = open(month_path,"a+")
                elif self.machine == "ccv":
                    sent_monthly = []
            else:
                first_to_do = timelist_original[int_counter]

            for index in range(timelist_original[-1]): # for every post

                if index < first_to_do:
                    pass
                else:
                    if index == first_to_do:
                        int_counter += 1
                        yr,mo = dates[int_counter]
                        month_path = self.model_path + "/c_sentiments/c_sentiments-{}-{}".format(yr,mo)
                        if not Path(month_path).is_file():
                            if self.machine == "local":
                                sent_monthly = open(month_path,"a+")
                            elif self.machine == "ccv":
                                with open(month_path,"w") as monthly:
                                    for post in sent_monthly:
                                        print(post,file=monthly)
                                sent_monthly = []
                        else: # if the monthly file exists on disk
                            first_to_do = timelist_original[int_counter]
                            sent_monthly = []

                    if index >= first_to_do:
                        if self.machine == "local":
                            print(c_sentiments[index],file=sent_monthly)
                        elif self.machine == "ccv":
                            sent_monthly.append(c_sentiments[index])

            int_counter = 0
            yr,mo = dates[int_counter]
            month_path = self.model_path + "/sentiments/sentiments-{}-{}".format(yr,mo)

            if not Path(month_path).is_file():
                if self.machine == "local":
                    sent_monthly = open(month_path,"a+")
                elif self.machine == "ccv":
                    sent_monthly = []
            else:
                first_to_do = timelist_original[int_counter]

            for index in range(timelist_original[-1]): # for every post

                if index < first_to_do:
                    pass
                else:
                    if index == first_to_do:
                        int_counter += 1
                        yr,mo = dates[int_counter]
                        month_path = self.model_path + "/sentiments/sentiments-{}-{}".format(yr,mo)
                        if not Path(month_path).is_file():
                            if self.machine == "local":
                                sent_monthly = open(month_path,"a+")
                            elif self.machine == "ccv":
                                with open(month_path,"w") as monthly:
                                    for post in sent_monthly:
                                        print(post,file=monthly)
                                sent_monthly = []
                        else: # if the monthly file exists on disk
                            first_to_do = timelist_original[int_counter]
                            sent_monthly = []

                    if index >= first_to_do:
                        if self.machine == "local":
                            print(sentiments[index],file=sent_monthly)
                        elif self.machine == "ccv":
                            sent_monthly.append(sentiments[index])


    ## Determines what percentage of the posts in each year was relevant based
    # on content filters

    # NOTE: Requires total comment counts (RC_Count_Total) from parser or disk

    # NOTE: Requires monthly relevant counts from parser or disk

    def Rel_Counter(self):
        if not Path(self.model_path + "counts/RC_Count_List").is_file():
            raise Exception(
                'Cumulative monthly comment counts could not be found')
        if not Path(self.model_path + "total_count/total_count").is_file():
            raise Exception(
                'Total monthly counts could not be found')

        # load the total monthly counts into a list
        monthly_list = []
        with open(self.model_path + "total_count/total_count", 'r') as f:
            for line in f:
                line = line.replace("\n", "")
                if line.strip() != "":
                    monthly_list.append(line)

        d = {}
        for idx, tuples in enumerate(self.dates):
            d[tuples] = int(monthly_list[idx])

        # calculate the total yearly counts
        total_year = {}
        for keys in d:
            if str(keys[0]) in total_year:
                total_year[str(keys[0])] += d[keys]
            else:
                total_year[str(keys[0])] = d[keys]

        relevant_year, _ = Get_Counts(self.model_path, frequency="yearly")

        relevant = {}
        for idx, year in enumerate(relevant_year):
            relevant[str(self.dates[0][0] + idx)] = year

        # calculate the percentage of comments in each year that was relevant and write it to file
        perc_rel = {}
        rel = open(self.model_path + "/perc_rel", 'a+')
        for key in relevant:
            perc_rel[key] = float(relevant[key]) / float(total_year[key])
        print(sorted(perc_rel.items()), file=rel)
        rel.close

    ## Load, calculate or re-calculate the percentage of relevant comments/year
    def Perc_Rel_RC_Comment(self):
        if Path(self.model_path + "/perc_rel").is_file():  # look for extant record
            # if it exists, ask if it should be overwritten
            Q = input(
                "Yearly relevant percentages are already available. Do you wish to delete them and count again [Y/N]?")

            if Q == 'Y' or Q == 'y':  # if yes
                os.remove(model_path + "/perc_rel")  # delete previous record
                self.Rel_Counter()  # calculate again
            else:  # if no
                print("Operation aborted")  # pass

        else:  # if there is not previous record
            self.Rel_Counter()  # calculate

    # Helper functions for select_random_comments
    def get_comment_lengths(self):
        fin = self.model_path + 'lda_prep/lda_prep'
        with open(fin, 'r') as fh:
            return [len(line.split()) for line in fh.read().split("\n")]

    def _select_n(self, n, iterable):
        if len(iterable) < n:
            return iterable
        return np.random.choice(iterable, size=n, replace=False)

    ## Selects a random subset of comments from the corpus to analyze

    # Parameters:
    #   n: Number of random comments to sample.
    #   years_to_sample: Years to select from.
    #   min_n_comments: Combine all comments from years with less than
    #       min_n_comments comments and select from the combined set. E.g. If
    #       there are fewer comments than (n) from a certain year, a random
    #       sample of n will be drawn from the pooled set of relevant comments
    #       from as many consecutive years as needed for the pool size to exceed
    #       (n).
    #   overwrite: If the sample file for the year exists, skip.
    def select_random_comments(self, n=n_random_comments,
                               years_to_sample=years, min_n_comments=5000,
                               overwrite=OVERWRITE):
        fns = self.get_parser_fns()
        fns["indices_random"] = "{}/random_indices".format(self.model_path)
        fns["counts_random"] = "{}/Random_Count_List".format(self.model_path)
        fns["timedict_random"] = "{}/Random_Count_Dict".format(self.model_path)
        fout = fns["indices_random"]

        if (not overwrite and os.path.exists(fout)):
            print("{} exists. Skipping. Set overwrite to True to overwrite.".format(fout))
            return

        years_to_sample = sorted(years_to_sample)
        ct_peryear, ct_cumyear = Get_Counts(path=self.model_path, frequency="yearly")
        ct_permonth, ct_cummonth = Get_Counts(
            path=self.model_path, frequency="monthly")
        assert len(self.dates) == len(ct_permonth) == len(ct_cummonth)
        ct_lu_by_year = dict((y, i) for i, y in enumerate(years))
        ct_lu_by_month = dict(zip(self.dates, range(len(self.dates))))
        early_years = [yr for yr in years_to_sample if
                       ct_peryear[ct_lu_by_year[yr]] < min_n_comments]

        # Make sure the early_years actually contains the first years in years, if
        # any. Otherwise the order that indices are written to file won't make any
        # sense.
        assert all([early_years[i] == early_years[i - 1] + 1 for i in range(1,
                                                                            len(early_years))])
        assert all([yr == yr_ for yr, yr_ in zip(early_years,
                                                 years_to_sample[:len(early_years)])])

        later_years = [yr for yr in years_to_sample if yr not in early_years]

        # Record the number of indices sampled per month
        nixs = defaultdict(int)

        # Get a list of comment lengths, so we can filter by it
        lens = self.get_comment_lengths()

        with open(fout, 'w') as wfh:
            if len(early_years) > 0:
                fyear, lyear = early_years[0], early_years[-1]
                if fyear - 1 in ct_lu_by_year:
                    start = ct_cumyear[ct_lu_by_year[fyear - 1]]
                else:
                    start = 0
                end = ct_cumyear[ct_lu_by_year[lyear]]
                ixs_longenough = [ix for ix in range(start, end) if lens[ix] >=
                                  min_comm_length]
                ixs = sorted(self._select_n(n, ixs_longenough))
                for ix in ixs:
                    nixs[self.dates[[ct > ix for ct in ct_cummonth].index(True)]] += 1
                assert sum(nixs.values()) == len(ixs)
                wfh.write('\n'.join(map(str, ixs)))
                wfh.write('\n')
            for year in later_years:
                start = ct_cumyear[ct_lu_by_year[year - 1]]
                end = ct_cumyear[ct_lu_by_year[year]]
                ixs_longenough = [ix for ix in range(start, end) if lens[ix] >=
                                  min_comm_length]
                ixs = sorted(self._select_n(n, ixs_longenough))
                nixs[year] = len(ixs)
                wfh.write('\n'.join(map(str, ixs)))
                wfh.write('\n')

        with open(fns["timedict_random"], "w") as tdfh:
            with open(fns["counts_random"], "w") as cfh:
                cumul_docs = 0
                for date in self.dates:
                    docs = nixs[date]
                    month = format_date(*date)
                    print(month + " " + str(docs), end='\n', file=tdfh)
                    # Use timedict data to populate counts file
                    cumul_docs += docs
                    print(cumul_docs, end='\n', file=cfh)
