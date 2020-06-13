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
    if Path(path + '/' + proper_filename).is_file():
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
                 download_raw=DOWNLOAD_RAW, hashsums=None, NN=NN, path=path,
                 legality=legality, marijuana=marijuana, stop=stop,
                 write_original=WRITE_ORIGINAL, vote_counting=vote_counting, author=author, sentiment=sentiment,
                 on_file=on_file,num_process=num_process,rel_sample_num=rel_sample_num,balanced_rel_sample=balanced_rel_sample,Neural_Relevance_Filtering=Neural_Relevance_Filtering):
        # check input arguments for valid type
        assert type(vote_counting) is bool
        assert type(author) is bool
        assert type(sentiment) is bool
        assert type(NN) is bool
        assert type(write_original) is bool
        assert type(download_raw) is bool
        assert type(clean_raw) is bool
        assert type(path) is str
        # check the given path
        if not os.path.exists(path):
            raise Exception('Invalid path')
        assert type(stop) is set or type(stop) is list

        self.clean_raw = CLEAN_RAW
        self.dates = dates
        self.download_raw = download_raw
        self.hashsums = hashsums
        self.NN = NN
        self.path = path
        self.legality = legality
        self.marijuana = marijuana
        self.stop = stop
        self.write_original = write_original
        self.vote_counting = vote_counting
        self.author = author
        self.sentiment = sentiment
        self.on_file = on_file
        self.bert_tokenizer = bert_tokenizer
         # connect the Python wrapper to the server
        # Instantiate CoreNLP wrapper than can be used across multiple threads
        self.nlp_wrapper = nlp_wrapper
        self.num_process = num_process
        self.rel_sample_num = rel_sample_num
        self.balanced_rel_sample = balanced_rel_sample
        self.Neural_Relevance_Filtering = Neural_Relevance_Filtering
        nltk.download('vader_lexicon')

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
        os.system('cd {} && wget {}'.format(self.path, url))

    ## Get Reddit compressed data file hashsums to check downloaded files'
    # integrity
    def Get_Hashsums(self):
        # notify the user
        print('Retrieving hashsums to check file integrity')
        # set the URL to download hashsums from
        url = 'https://files.pushshift.io/reddit/comments/sha256sum.txt'
        # remove any old hashsum file
        if Path(self.path + '/sha256sum.txt').is_file():
            os.remove(self.path + '/sha256sum.txt')
        # download hashsums
        os.system('cd {} && wget {}'.format(self.path, url))
        # retrieve the correct hashsums
        hashsums = {}
        with open(self.path + '/sha256sum.txt', 'rb') as f:
            for line in f:
                line = line.decode("utf-8")
                if line.strip() != "":
                    (val, key) = str(line).split()
                    hashsums[key] = val
        return hashsums

    ## calculate hashsums for downloaded files in chunks of size 4096B
    def sha256(self, fname):
        hash_sha256 = hashlib.sha256()
        with open("{}/{}".format(self.path, fname), "rb") as f:
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
            'attention_masks' : attention_masks
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
        fns = dict((("original_comm", "{}/original_comm/original_comm{}".format(self.path, suffix)),
                    ("original_indices", "{}/original_indices/original_indices{}".format(self.path, suffix)),
                    ("counts", "{}/counts/RC_Count_List{}".format(self.path, suffix)),
                    ("timedict", "{}/timedict/RC_Count_Dict{}".format(self.path, suffix)),
                    ("total_count", "{}/total_count/total_count{}".format(self.path, suffix))
                    ))
        if self.NN:
            # fns["nn_prep"] = "{}/nn_prep/nn_prep{}".format(self.path, suffix)
            fns["bert_prep"] = "{}/bert_prep/bert_prep{}.json".format(self.path, suffix)
        else:
            fns["lda_prep"] = "{}/lda_prep/lda_prep{}".format(self.path, suffix)
        if self.vote_counting:
            fns["votes"] = "{}/votes/votes{}".format(self.path, suffix)
        if self.author:
            fns["author"] = "{}/author/author{}".format(self.path, suffix)
        if self.sentiment:
            fns["sentiments"] = "{}/sentiments/sentiments{}".format(self.path, suffix)
            fns["c_sentiments"] = "{}/sentiments/c_sentiments{}".format(self.path, suffix)
            fns["t_sentiments"] = "{}/sentiments/t_sentiments{}".format(self.path, suffix)
            fns["v_sentiments"] = "{}/sentiments/v_sentiments{}".format(self.path, suffix)
        return fns

    ## Receives as input one document and its index, as well as output address
    # writes sentiment values derived from 3 packages separately to file, averages
    # them and stores the average as well
    def write_avg_sentiment(self, original_body, sentiments,month,main_counter,fns):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        tokenized = sent_detector.tokenize(original_body)

        total_vader = 0
        total_core_nlp = 0
        total_textblob = 0
        original_without_quotes = original_body.replace("\"", "")

        try:
            annot_doc = self.nlp_wrapper.annotate(original_without_quotes, properties={
                'annotators': 'sentiment',
                'outputFormat': 'json',
                'timeout': 1000000 })
            with open(fns["c_sentiments"],"a+") as c_sentiments:
                if len(annot_doc['sentences']) == 0:
                    c_sentiments.write(str(0)+",\n")
                else:
                    for i in range(0, len(annot_doc['sentences'])):
                        total_core_nlp += int(annot_doc['sentences'][i]['sentimentValue'])
                        c_sentiments.write(str(annot_doc['sentences'][i]['sentimentValue'])+",")
                    c_sentiments.write("\n")
        except:
            print("CoreNLP error")
            with open("CoreNLP_errors.txt","a+") as errors:
                errors.write(str(month)+","+str(main_counter)+","+ " ".join(original_body.split()).replace("\n",""))
            # potential BUG: do we need to write None to the sentiment file? Or does the failed sentence just get ignored?

        with open(fns["v_sentiments"],"a+") as v_sentiments, open(fns["t_sentiments"],"a+") as t_sentiments:
            for sentence in tokenized:
                # Vader score
                sid = SentimentIntensityAnalyzer()
                score_dict = sid.polarity_scores(sentence)
                total_vader += score_dict['compound']
                v_sentiments.write(str(score_dict['compound'])+",")

                # Get TextBlob sentiment
                blob = TextBlob(sentence)
                total_textblob += blob.sentiment[0]
                t_sentiments.write(str(blob.sentiment[0])+",")
            v_sentiments.write("\n")
            t_sentiments.write("\n")
        avg_vader = total_vader / len(tokenized)
        avg_blob = total_textblob / len(tokenized)
        avg_core_nlp = total_core_nlp / len(annot_doc['sentences'])
        # Normalizing core nlp so it's between -1 and 1
        normalized_core_nlp = ((avg_core_nlp / 4) * 2) - 1
        avg_score = (avg_vader + avg_blob + normalized_core_nlp) / 3
        print(avg_score, file=sentiments)

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

            print("The following needed processed file(s) were missing for "
                  + str(year) + ", month " + str(month) + ":")
            print(missing_parsing_files)
            print("Initiating preprocessing of " + filename + "at  "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))


            # preprocess raw data
            # if the file is available on disk and download is on, prevent deletion
            if not filename in self.on_file and self.download_raw:
                print("MONTH and YEAR")
                print(month)
                print(year)
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
            if self.NN:  # if doing NN
                fout = open(fns["bert_prep"], 'w')
            else:  # if doing LDA
                fout = open(fns["lda_prep"], 'w')

            # create a file if we want to write the original comments and their indices to disk
            if self.write_original:
                foriginal = open(fns["original_comm"], 'w')
                main_indices = open(fns["original_indices"], 'w')

            # if we want to record the votes
            if self.vote_counting:
                # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
                vote = open(fns["votes"], 'w')

            # if we want to record the author
            if self.author:
                # create a file for storing whether a relevant comment has been upvoted or downvoted more often or neither
                author = open(fns["author"], 'w')

            if self.sentiment:
                # create a file for storing the average sentiment of a post
                sentiments = open(fns["sentiments"], 'w')
                v_sentiments = open(fns["v_sentiments"], 'w')
                t_sentiments = open(fns["t_sentiments"], 'w')
                c_sentiments = open(fns["c_sentiments"], 'w')

            # create a file to store the relevant cummulative indices for each month
            ccount = open(fns["counts"], 'w')

            main_counter = 0

            # open the file as a text file, in utf8 encoding, based on encoding type
            if '.zst' in filename:
                file = open(filename, 'rb')
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(file)
                fin = io.TextIOWrapper(stream_reader, encoding='utf-8')
            elif '.xz' in filename:
                fin = lzma.open(filename, 'r')
            elif '.bz2' in filename:
                fin = bz2.BZ2File(self.path + '/' + filename, 'r')
            else:
                raise Exception('File format not recognized')

            # read data
            per_file_counter = 0
            for line in fin:  # for each comment
                main_counter += 1  # update the general counter

                # decode and parse the json, and turn it into regular text
                if not '.zst' in filename:
                    comment = line.decode()
                else:
                    comment = line
                comment = decoder.decode(comment)
                original_body = html.unescape(comment["body"])  # original text

                # filter comments by relevance to the topic according to regex
                if any(not exp.search(original_body.lower()) is None for exp in marijuana) and any(
                        not exp.search(original_body.lower()) is None for exp in legality):

                    # preprocess the comments
                    ## TODO is this for one comment or for all of the comments in a month?
                    if self.NN:
                        # Tokenize the sentences
                        #body = sent_detector.tokenize(
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
                        body = self._clean(original_body).lower()

                    else:  # if doing LDA

                        # clean the text for LDA
                        body = self.LDA_clean(original_body)

                    if body.strip() == "":  # if the comment is not empty after preprocessing
                        pass
                    else:
                        # remove mid-comment lines
                        body = body.replace("\n", "")
                        body = " ".join(body.split())

                        # print the comment to file
                        print(body, sep=" ", end="\n", file=fout)

                        # If calculating sentiment, write the average sentiment to
                        # file. Range is -1 to 1, with values below 0 meaning neg
                        # sentiment
                        if self.sentiment:
                            self.write_avg_sentiment(original_body, sentiments,month,main_counter,fns)
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
                    print(main_counter, end="\n", file=counter_file)

            # close the files to save the data
            fin.close()
            fout.close()
            if self.vote_counting:
                vote.close()
            if self.write_original:
                foriginal.close()
                main_indices.close()
            if self.author:
                author.close()
            if self.sentiment:
                sentiments.close()
                v_sentiments.close()
                c_sentiments.close()
                t_sentiments.close()
            ccount.write(str(per_file_counter)+"\n")
            ccount.close()
            with open(fns["timedict"], "wb") as wfh:
                pickle.dump(timedict, wfh)

        # reset the missing files list for the next month
        missing_parsing_files = []

        # timer
        print("Finished parsing " + filename + " at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        # if the user wishes compressed data files to be removed after processing
        if self.clean_raw and filename not in self.on_file and Path(filename).is_file():
            print("Cleaning up {}/{}.".format(self.path, filename))
            # delete the recently processed file
            os.system('cd {} && rm {}'.format(self.path, filename))

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

        # Parallelize parsing by month
        # NOTE: For best results, set the number of processes in the following
        # line based on (number of physical cores)*(hyper-threading multiplier)
        # -1 (for synchronization overhead)
        pool = multiprocessing.Pool(processes=num_process)
        inputs = [(year, month, self.on_file, self.__dict__) for year, month in self.dates]
        pool.map(parse_one_month_wrapper, inputs)
        # timer
        print("Finished parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))

        # Pool parsing data from all files
        self.pool_parsing_data()

    ## Function to safely create folder structure for parsed files
    def safe_dir_create(self):
        fns = self.get_parser_fns()
        for key in fns:
            try:
                new_path = os.path.join(self.path, key)
                os.makedirs(new_path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    continue
                else:
                    raise

    ## Function to call parser when needed and parse comments
    # TODO: Replace mentions of Vote in this file with mentions of sample_ratings
    # TODO: Add main counter and original comments and indices to this function
    def Parse_Rel_RC_Comments(self,num_process=num_process):
        # if preprocessed comments are available, ask if they should be rewritten
        if (self.NN and Path(self.path + "/bert_prep/bert_prep").is_file()) or (
                not self.NN and Path(self.path + "/lda_prep/lda_prep").is_file()):
            Q = input(
                "Preprocessed comments are already available. Do you wish to delete them and parse again [Y/N]?")
            if Q == 'Y' or Q == 'y':  # if the user wishes to overwrite the comments
                # delete previous preprocessed data
                if self.NN:  # for NN
                    shutil.rmtree(self.path + "/bert_prep")
                elif not self.NN:  # for LDA
                    shutil.rmtree(self.path + "/lda_prep")
                if Path(self.path + "/original_indices/original_indices").is_file() and self.write_original:
                    shutil.rmtree(self.path + "/original_indices")
                if Path(self.path + "/original_comm/original_comm").is_file() and self.write_original:
                    shutil.rmtree(self.path + "/original_comm")
                if Path(self.path + "/votes/votes").is_file() and self.vote_counting:
                    shutil.rmtree(self.path + "/votes")
                if Path(self.path + "/author/author").is_file() and self.author:
                    shutil.rmtree(self.path + "/author")
                if Path(self.path + "/sentiments/sentiments").is_file() and self.sentiment:
                    shutil.rmtree(self.path + "/sentiments")
                if Path(self.path + "counts/RC_Count_List").is_file():
                    shutil.rmtree(self.path + "/counts")
                if Path(self.path + "timedict/RC_Count_Dict").is_file():
                    shutil.rmtree(self.path + "timedict")

                # timer
                print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
                self.parse(num_process)

            else:  # if preprocessed comments are available and
                # the user does not wish to overwrite them
                print("Checking for missing files")

                # check for other required files aside from main data
                missing_files = 0

                if not Path(self.path + "counts/RC_Count_List").is_file():
                    missing_files += 1

                if not Path(self.path + "votes/votes").is_file() and self.vote_counting:
                    missing_files += 1

                if not Path(self.path + "author/author").is_file() and self.author:
                    missing_files += 1

                if not Path(self.path + "sentiments/sentiments").is_file() and self.sentiment:
                    missing_files += 1

                if not Path(self.path + "sentiments/v_sentiments").is_file() and self.sentiment:
                    missing_files += 1

                if not Path(self.path + "sentiments/c_sentiments").is_file() and self.sentiment:
                    missing_files += 1

                if not Path(self.path + "sentiments/t_sentiments").is_file() and self.sentiment:
                    missing_files += 1

                # if there are missing files, delete any partial record and parse again
                if missing_files != 0:
                    print("Deleting partial record and parsing again")

                    if Path(self.path + "votes/votes").is_file():
                        shutil.rmtree(self.path + "votes")

                    if Path(self.path + "author/author").is_file():
                        shutil.rmtree(self.path + "author")

                    if Path(self.path + "sentiments/sentiments").is_file():
                        shutil.rmtree(self.path + "sentiments")

                    if self.NN:  # for NN
                        shutil.rmtree(self.path + "bert_prep")

                    elif not self.NN:  # for LDA
                        shutil.rmtree(self.path + "lda_prep")

                    if Path(self.path + "counts/RC_Count_List").is_file():
                        shutil.rmtree(self.path + "counts")

                    if Path(self.path + "timedict/RC_Count_Dict").is_file():
                        shutil.rmtree(self.path + "timedict")

                    # timer
                    print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
                    self.parse(num_process)

        else:
            if Path(self.path + "counts/RC_Count_List").is_file():
                shutil.rmtree(self.path + "counts")
            if Path(self.path + "votes/votes").is_file() and self.vote_counting:
                shutil.rmtree(self.path + "votes")
            if Path(self.path + "author/author").is_file() and self.author:
                shutil.rmtree(self.path + "author")
            if Path(self.path + "sentiments/sentiments").is_file() and self.sentiment:
                shutil.rmtree(self.path + "sentiments")
            if Path(self.path + "original_comm/original_comm").is_file() and self.write_original:
                shutil.rmtree(self.path + "original_comm")
            if Path(self.path + "original_indices/original_indices").is_file() and self.write_original:
                shutil.rmtree(self.path + "original_indices")

            # timer
            print("Started parsing at " + time.strftime('%l:%M%p, %m/%d/%Y'))
            self.parse(num_process)

    ## Function for removing non-English posts picked up by the regex filter
    def lang_filtering(self,dates=dates):

        if Path(self.path + "/non_en").is_file():  # if corpus is already filtered
            print("Found language filtering results on file. Moving on.")
            pass
        else:  # otherwise

            # check for msising files per parameter configs

            # raw dataset
            if not Path(self.path + "/original_comm/original_comm").is_file():
                raise Exception('Original comments could not be found')
            if not Path(self.path + "/original_indices/original_indices").is_file():
                raise Exception('Original indices could not be found')

            # preprocessed data
            if (not Path(self.path + "/lda_prep/lda_prep").is_file()) and self.NN == False:
                raise Exception('Preprocessed dataset could not be found')
            elif (not Path(self.path + "/bert_prep/bert_prep.json").is_file()) and self.NN == True:
                raise Exception('Preprocessed dataset could not be found')

            # cumulative post counts
            if not Path(self.path + "/counts/RC_Count_List").is_file():
                raise Exception(
                    'Cumulative monthly comment counts could not be found')
            else:  # load the cumulative counts
                timelist_original = []
                with open(self.path + "/counts/RC_Count_List", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            timelist_original.append(int(float(line.strip())))

            # post meta-data
            if (not Path(self.path + "/votes/votes").is_file()) and self.vote_counting:
                raise Exception('Votes counld not be found')
            if (not Path(self.path + "/author/author").is_file()) and self.author:
                raise Exception('Author usernames could not be found')
            if (not Path(self.path + "/sentiments/sentiments").is_file()) and self.sentiment:
                raise Exception('Sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/v_sentiments").is_file()) and self.sentiment:
                raise Exception('Vader sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/t_sentiments").is_file()) and self.sentiment:
                raise Exception('TextBlob sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/c_sentiments").is_file()) and self.sentiment:
                raise Exception('CoreNLP sentiment estimates could not be found')
            if (not Path(self.path + "/original_comm/auto_labels").is_file()) and self.Neural_Relevance_Filtering:
                raise Exception('Auto-generated relevance labels could not be found')

            # Initialize variables

            # timer
            print("Started filtering out non-English posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))

            # seed the random initializer
            DetectorFactory.seed = 0

            # counters for the number of non-English posts from each time period
            int_non_en = np.zeros(len(timelist_original)+1)

            non_en_idx = []  # list for indices of non-English posts

            int_counter = 0  # counter for the time period an index belongs to

            # Filter the posts

            with open(self.path + "/non_en", "w") as non_en:
                total_count = 0
                with open(self.path + "/original_comm/original_comm", "r") as raw_dataset:
                    for index, post in enumerate(raw_dataset):
                        total_count += 1
                        try:
                            if index > timelist_original[int_counter]:
                                int_counter += 1  # update time interval counter
                        except:
                            int_counter += 1
                        try:  # if post is too short to reliably analyze or
                            # highly likely to be in English
                            if detect(post) == 'en' or len(post.split()) <= 20:
                                pass
                            else:  # if post is likely not to be in English
                                non_en_idx.append(index)  # record the index
                                int_non_en[int_counter] += 1  # update
                                # non-English post counter
                                print(index, end="\n", file=non_en)  # save index
                        except:  # if language identification failed, add the
                            # post to the list of posts to be removed from dataset
                            non_en_idx.append(index)  # record the index
                            int_non_en[int_counter] += 1  # update non-English
                            # post counter
                            print(index, end="\n", file=non_en)  # save index

            # remove the marked posts from the raw dataset
            with open(self.path + "/original_comm/original_comm", "r") as f:
                lines = f.readlines()
            with open(self.path + "/original_comm/original_comm", "w") as f:
                for index, line in enumerate(lines):
                    if line.strip() != "" and index not in non_en_idx:
                        f.write(line)

            with open(self.path + "/original_indices/original_indices", "r") as f:
                lines = f.readlines()
            with open(self.path + "/original_indices/original_indices", "w") as f:
                for index, line in enumerate(lines):
                    if line.strip() != "" and index not in non_en_idx:
                        f.write(line)

            if self.Neural_Relevance_Filtering and Path(self.path + "/original_comm/auto_labels").is_file():
                with open(self.path + "/original_comm/auto_labels", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/original_comm/auto_labels", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

            #BUG: bert_prep is not being updated. Okay for now, but bad if we'll be using it

            # remove the marked posts from the processed dataset
            if self.NN == True:
                # with open(self.path + "/bert_prep/bert_prep.json", "r") as f:
                #     lines = f.readlines()
                # with open(self.path + "/bert_prep/bert_prep.json", "w") as f:
                #     for index, line in enumerate(lines):
                #         if line.strip() != "" and index not in non_en_idx:
                #             f.write(line)
                pass
            else:
                with open(self.path + "/lda_prep/lda_prep", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/lda_prep/lda_prep", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

            # correct post counts for all time intervals

            running_tot_count = 0
            for interval, count in enumerate(timelist_original):
                running_tot_count += int_non_en[interval]
                timelist_original[interval] = timelist_original[interval] - running_tot_count

            with open(self.path + "/counts/RC_Count_List", "w") as f:
                for interval in timelist_original:
                    print(interval, end="\n", file=f)

            # TODO: debug
            total_counter = 0

            for yr,mo in dates:
                file_counter = 0
                interval_counter = 0
                with open(self.path+"/counts/RC_Count_List-{}-{}".format(yr,mo),"r") as f:
                    for line in f:
                        if line.strip() != "":
                            file_counter = int(line.strip())

                for index in non_en_idx:
                    if index >= total_counter and index < total_counter + file_counter:
                        interval_counter += 1
                total_counter += int(line.strip())
                with open(self.path+"/counts/RC_Count_List-{}-{}".format(yr,mo),"w") as f:
                    new_count = file_counter - interval_counter
                    f.write(str(new_count))

            # remove appended information (if any) related to marked posts
            if self.vote_counting:
                with open(self.path + "/votes/votes", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/votes/votes", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

            if self.author:
                with open(self.path + "/author/author", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/author/author", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

            if self.sentiment:
                with open(self.path + "/sentiments/sentiments", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/sentiments/sentiments", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)
                with open(self.path + "/sentiments/v_sentiments", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/sentiments/v_sentiments", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)
                with open(self.path + "/sentiments/t_sentiments", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/sentiments/t_sentiments", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)
                with open(self.path + "/sentiments/c_sentiments", "r") as f:
                    lines = f.readlines()
                with open(self.path + "/sentiments/c_sentiments", "w") as f:
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in non_en_idx:
                            f.write(line)

            # timer
            print("Finished filtering out non-English posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))

    ## OBSOLETE function: Per-sentence TextBlob, Vader and CoreNLP sentiment est.
    # for the entire corpus

    # NOTE: A post is positive if Vader estimate >0.05, negative if it is <-0.05
    # neutral otherwise.
    # NOTE: For CoreNLP, value bindings are: 0=Very negative,1=Negative,
    # 2=Neutral, 3=Positive, 4=Very Positive. Averaging across sentences may
    # result in non-integer values

    # NOTE: Since the retrival of values is slow for some packages and the
    # connection might time out, this function allows for resuming retrieval
    # by ignoring the comments analyzed so far. This is done automatically by
    # first counting the number of lines in files on disc and comparing that
    # to the size of the corpus.

    # NOTE: Always mark and check the last processed post in the sentiment files
    # to make sure that duplicate sentiment values are not being stored.

    def add_sentiment(self):

        # Retrieve the total number of posts in the corpus (needs RC_Count_List)
        if not Path(self.path + "/RC_Count_List").is_file():
            raise Exception(
                'Cumulative monthly comment counts could not be found')

        with open(self.path + "counts/RC_Count_List", 'r') as f:
            timelist = []
            for line in f:
                if line.strip() != "":
                    timelist.append(int(line))
        goal = timelist[-1]

        # Vader estimates

        # check to see if Vader estimates already exist
        # If not available, start from the first comment
        if not Path(self.path + "/sentiments_vader").is_file():
            ignore_v = 0
        else:  # otherwise, examine how many comments were analyzed
            ignore_v = 0
            with open(self.path + "/sentiments_vader", 'r') as f:
                for comment in f:
                    ignore_v += 1

        # CoreNLP estimates

        # If estimates are not available, start from the first comment
        if not Path(self.path + "/sentiments_core").is_file():
            ignore_c = 0
        else:  # otherwise, examine how many comments were analyzed
            ignore_c = 0
            with open(self.path + "/sentiments_core", 'r') as f:
                for comment in f:
                    ignore_c += 1

        # TextBlob estimates

        # If estimates are not available, start from the first comment
        if not Path(self.path + "/sentiments_blob").is_file():
            ignore_t = 0
        else:  # otherwise, examine how many comments were analyzed
            ignore_t = 0
            with open(self.path + "/sentiments_blob", 'r') as f:
                for comment in f:
                    ignore_t += 1

        if ignore_v == goal and ignore_c == goal and ignore_t == goal:
            # if all estimates for all of the comments exist, move along
            pass
        else:  # otherwise, retrieve estimates from from the packages

            # check to see if corpus text is accessible
            if not Path(self.path + "/original_comm").is_file():
                raise Exception('Original comments could not be found')

            print("Started retrieving and recording sentiment values at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))  # timer

            print("Vader estimates exist for the first " + str(ignore_v) + " posts. Ignoring.")
            print("TextBlob estimates exist for the first " + str(ignore_t) + " posts. Ignoring.")
            print("CoreNLP estimates exist for the first " + str(ignore_c) + " posts. Ignoring.")

            # set up various packages

            # Vader
            sid = SentimentIntensityAnalyzer()  # Vader sentiment analyzer object

            # CoreNLP
            # create a connection to the CoreNLP server to retrieve sentiment
            # (requires CoreNLP_server.py in the same directory)
            subprocess.run(['gnome-terminal -x python CoreNLP_server.py'], shell=True)
            time.sleep(5)  # wait for connection to the server to be established
            # connect the Python wrapper to the server
            nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

            # read the corpus and retrieve compound Vader score for each post
            with open(self.path + "/original_comm", "r") as texts, \
                    open(self.path + "/sentiments_vader", "a+") as vader, \
                    open(self.path + "/sentiments_blob", "a+") as textblob, \
                    open(self.path + "/sentiments_core", "a+") as core:

                text_reader = texts.readlines()  # set up reader

                # Prepare sentence tokenizer
                sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

                for index, comment in enumerate(text_reader):

                    tokenized = sent_detector.tokenize(comment)

                    if index >= ignore_v:

                        sentence_vals = []

                        for sentence in tokenized:
                            ss = sid.polarity_scores(sentence)
                            sentence_vals.append(ss["compound"])

                        print(",".join([str(sentence) for sentence in sentence_vals]), end='\n', file=vader)

                    if index >= ignore_t:

                        sentence_vals = []

                        for sentence in tokenized:
                            blob = TextBlob(sentence)
                            sentence_vals.append(blob.sentiment[0])

                        print(",".join([str(sentence) for sentence in sentence_vals]), end='\n', file=textblob)

                    if index >= ignore_c:

                        # retrieve sentiment estimates from CoreNLP
                        annot_doc = nlp_wrapper.annotate(comment, properties={
                            'annotators': 'sentiment',
                            'outputFormat': 'json',
                            'timeout': 1000000, })

                        # store the values for each sentence in a list
                        sent_values = []

                        for sentence in annot_doc["sentences"]:
                            sent_values.append(sentence["sentimentValue"])

                        # write the average sentiment value to disk
                        print(",".join([str(sentence) for sentence in sent_values]), end='\n', file=core)

                    if index != 0 and (index + 1) % 1000 == 0:  # every 1000 comments
                        texts.flush()  # flush the results to disk

        print("Finished retrieving and recording sentiment values at "
              + time.strftime('%l:%M%p, %m/%d/%Y'))  # timer

    ## Uses a pre-trained neural network to prune dataset from irrelevant posts
    def Neural_Relevance_Screen(self,data_path=path+"/original_comm/",
    model_path=path+"/Human_Ratings/1_1/full_1005/",dates=dates,
    rel_sample_num=rel_sample_num,balanced_rel_sample=balanced_rel_sample):

        # check for previous screening results
        if Path(data_path + "sample_auto_labeled.csv").is_file():

            print("A sample of auto-labeled posts was found, suggesting neural relevance screening was previously performed. Moving on.")

        else: # if screening results not found

            # Load cumulative number of relevant posts for each month, from disk

            if not Path(self.path + "/counts/RC_Count_List").is_file():
                raise Exception(
                    'Cumulative monthly comment counts could not be found')
            else:
                timelist_original = []
                with open(self.path + "/counts/RC_Count_List", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            timelist_original.append(int(line))

            # set up neural network runtime configurations
            logging.basicConfig(level=logging.INFO)
            transformers_logger = logging.getLogger("transformers")
            transformers_logger.setLevel(logging.WARNING)

            # load the pre-trained neural network from model_path
            model = ClassificationModel('roberta', model_path, use_cuda=False,
            args={'fp16': False,'num_train_epochs': 1, 'manual_seed':1, 'silent':True})

            total_count = 0 # counter for all documents in the dataset

            # check for a complete record of screening
            if Path(data_path+"auto_labels").is_file():
                print("Found full dataset labels. Loading.")

                with open(data_path+"auto_labels","r") as f:
                    for line in f:
                        if line.strip() != "":
                            total_count += 1

            else: # if no previous record is found

                # start a general record
                with open(data_path+"auto_labels","w") as general_labels:

                    for yr,mo in dates: # for each month

                        start = time.time() # measure processing time

                        # if labels for that month exist, load them
                        if Path(data_path+"auto_labels-{}-{}".format(yr,mo)).is_file():
                            print("Found labels for year "+str(yr)+", month "+str(mo)+". Loading.")

                            with open(data_path+"auto_labels-{}-{}".format(yr, mo),"r") as labels:
                                for label in labels:
                                    if label.strip() != "":
                                        general_labels.write(label)
                                        total_count += 1
                        else: # otherwise use the trained network to obtain them

                            # get the number of documents within each parsed monthly data file
                            # NOTE: Might be slightly different from the actual
                            # monthly counts
                            file_counter = 0
                            with open(self.path+"/counts/RC_Count_List-{}-{}".format(yr, mo),"r") as counts:
                                for line in counts:
                                    if line.strip() != "":
                                        file_counter = int(line)
                                        total_count += int(line)

                            # use original comment text to auto-generate labels
                            with open(data_path+"original_comm-{}-{}".format(yr, mo),"r") as texts, open(data_path+"auto_labels-{}-{}".format(yr, mo),"w") as labels:

                                counter = 0 # doc counter for batching

                                batch = []
                                for line in texts:

                                    batch.append(line)
                                    counter += 1

                                    if counter % 100 == 0 or counter == file_counter:

                                        prediction, raw_output = model.predict(batch)
                                        for element in prediction:
                                            labels.write(str(element)+"\n")
                                            general_labels.write(str(element)+"\n")

                                        batch = []

                        # calculate and report processing time
                        end = time.time()
                        hours, rem = divmod(end-start, 3600)
                        minutes, seconds = divmod(rem, 60)
                        print("Processed month {} of year {} in {:0>2}:{:0>2}:{:05.2f}".format(mo,yr,int(hours),int(minutes),seconds))

            # Sample rel_sample_num documents at random for evaluating the dataset and the auto-labeling
            random_sample = list(np.random.choice(range(0,total_count),size=rel_sample_num,replace=False))
            labels_array = np.ones(total_count) # initiate array for labels

            # read labels from disk and identify indices of irrelevant posts
            irrel_idxes = []
            with open(data_path+"auto_labels","r") as labels:
                for idx,line in enumerate(labels):
                    if line.strip() == '0' or line.strip() == 'None':
                        labels_array[idx] = 0
                        irrel_idxes.append(idx)

            # TODO: Add the ability to deal with posts for which the classifier
            # has failed to provide a prediction but rather an error (None classes).

            # if obtaining a sample of labels balanced across output categories:
            if balanced_rel_sample:
                sampled_cats = {0:0,1:0}
                goal = {0:floor(float(rel_sample_num) / 2),1:ceil(float(rel_sample_num) / 2)}

                for value in random_sample:
                    sampled_cats[int(labels_array[value])] += 1

                print("Random sample composition: ")
                print(sampled_cats)

                for category,_ in sampled_cats.items(): # upsample and downsample
                # to have a balance of posts marked positive or negative

                    while sampled_cats[category] < goal[category]:

                        if category == 0:
                            relevant_subset = [i for i in irrel_idxes if i not in random_sample]
                        elif category == 1:
                            relevant_subset = [i for i in range(0,total_count) if i not in irrel_idxes and i not in random_sample]

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
            print("Sampling "+str(len(set(random_sample)))+" documents")

            sampled_docs = []
            int_counter = 0

            with open(data_path+"original_comm","r") as sampling:
                sampled_idxes =[]
                for idx,sampler in enumerate(sampling):
                    try:
                        if idx > timelist_original[int_counter]:
                            int_counter += 1
                        if idx in random_sample:
                            # writing year, month, text to sample file
                            sampled_docs.append([str(dates[int_counter][0]),str(dates[int_counter][1]),sampler])
                    except: # if the date associated with the post is beyond
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

                            sampled_docs.append([str(yr),str(mo),sampler])

            # load labels for the sampled documents
            general_counter = 0
            sample_counter = 0
            with open(data_path+"auto_labels","r") as labels:
                for idx,line in enumerate(labels):
                    if idx in random_sample:
                        sampled_docs[sample_counter].append(line)
                        sample_counter += 1

            # shuffle the docs and check number and length
            np.random.shuffle(sampled_docs)
            assert len(sampled_docs) == rel_sample_num
            for element in sampled_docs:
                assert len(element) == 4

            # write the sampled files to a csvfile
            with open(data_path+"sample_auto_labeled.csv", 'a+') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['year','month','text','auto label','accuracy'])
                for document in sampled_docs:
                    writer.writerow(document)

    ## Function that uses the results of Neural_Relevance_Screen to remove posts
    # likely to be irrelevant from the dataset.
    def Neural_Relevance_Clean(self,path=path, dates=dates):

        # check to see if relevance filtering was previously performed
        if Path(self.path+"rel_clean_cert.txt").is_file():
            print("A certificate for a previously finished relevance filtering was found. Moving on.")

        else: # if not

            # check for needed but missing files per parameter configs
            # BUG: bert_prep is not being updated. Okay for now, but will be problematic
            # if we end up using it

            # raw dataset and indices
            if not Path(self.path + "/original_comm/original_comm").is_file():
                raise Exception('Original comments could not be found')
            if not Path(self.path + "/original_indices/original_indices").is_file():
                raise Exception('Original indices could not be found')

            # preprocessed data for LDA and NN
            if (not Path(self.path + "/lda_prep/lda_prep").is_file()) and self.NN == False:
                raise Exception('Preprocessed dataset could not be found')
            elif (not Path(self.path + "/bert_prep/bert_prep.json").is_file()) and self.NN == True:
                raise Exception('Preprocessed dataset could not be found')

            # cumulative post counts
            if not Path(self.path + "/counts/RC_Count_List").is_file():
                raise Exception(
                    'Cumulative monthly comment counts could not be found')
            else:  # load the cumulative counts
                timelist_original = []
                with open(self.path + "/counts/RC_Count_List", "r") as f:
                    for line in f:
                        if line.strip() != "":
                            timelist_original.append(int(line))

            # post meta-data
            if (not Path(self.path + "/votes/votes").is_file()) and self.vote_counting:
                raise Exception('Votes counld not be found')
            if (not Path(self.path + "/author/author").is_file()) and self.author:
                raise Exception('Author usernames could not be found')
            if (not Path(self.path + "/sentiments/sentiments").is_file()) and self.sentiment:
                raise Exception('Sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/v_sentiments").is_file()) and self.sentiment:
                raise Exception('Vader sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/t_sentiments").is_file()) and self.sentiment:
                raise Exception('TextBlob sentiment estimates could not be found')
            if (not Path(self.path + "/sentiments/c_sentiments").is_file()) and self.sentiment:
                raise Exception('CoreNLP sentiment estimates could not be found')

            # Initialize variables

            # timer
            print("Started filtering out irrelevant posts at "
                  + time.strftime('%l:%M%p, %m/%d/%Y'))
            start = time.time() # start a clock for processing time

            # load negative or unknown labels for the dataset from disk

            int_counter = 0  # counter for the time period an index belongs to

            # counters for the number of irrelevant posts from each time period
            int_non_rel = np.zeros(len(timelist_original)+1)

            # counters for the number of irrelevant posts from each file
            file_non_rel = np.zeros_like(timelist_original)

            general_counter = 0 # counter for all comments
            file_counter = 0 # counter for comments in a monthly file
            negative_labels = [] # list for negative labels

            for yr,mo in dates: # for each month
                with open(path+"/original_comm/auto_labels-{}-{}".format(yr,mo),"r") as labels:
                    for idx,line in enumerate(labels):
                        if line.strip() != "":

                            try:
                                if general_counter > timelist_original[int_counter]:
                                    int_counter += 1 # try to update month counter as needed
                                # if label was 0 or there was an error in labeling, remove
                                if line.strip() == "None" or line.strip() == "0":
                                    negative_labels.append(general_counter) # add index
                                    int_non_rel[int_counter] += 1  # update counter
                                    file_non_rel[file_counter] += 1 # update counter

                            except: # if the date of the post is out of bounds for the file,
                            # ignored the requirement to update the monthly counter
                                # if label was 0 or there was an error in labeling, remove
                                if line.strip() == "None" or line.strip() == "0":
                                    negative_labels.append(general_counter) # add index
                                    int_non_rel[int_counter+1] += 1  # update counter
                                    file_non_rel[file_counter] += 1 # update counter

                            general_counter += 1 # update

                file_counter += 1 # update

            # Filter the posts

            # BUG: I'm not filtering bert_prep. It's okay as long as we don't use it

            # A list of dataset files needing to be updated based on parameters
            filenames = ['/original_comm/original_comm','/original_indices/original_indices','/original_comm/auto_labels']
            if not self.NN:
                filenames.append("/lda_prep/lda_prep")
            if self.vote_counting:
                filenames.append("/votes/votes")
            if self.author:
                filenames.append("/author/author")
            if self.sentiment:
                filenames.append("/sentiments/sentiments")
                filenames.append("/sentiments/v_sentiments")
                filenames.append("/sentiments/c_sentiments")
                filenames.append("/sentiments/t_sentiments")

            for file in filenames: # for each file in the list above

                with open(self.path + file,"r") as f: # read each line
                    lines = f.readlines()
                with open(self.path + file, "w") as f: # write only the relevant posts
                    for index, line in enumerate(lines):
                        if line.strip() != "" and index not in negative_labels:
                            f.write(line)

            # update document counts for each time interval in the dataset
            running_tot_count = 0
            for interval, count in enumerate(timelist_original):
                running_tot_count += int_non_rel[interval]
                timelist_original[interval] = timelist_original[interval] - running_tot_count

            # get the actual post count (might be different from the list above
            # because of out-of-bounds posts, for which an additional entry will
            # be appended to the count list)
            with open(path+'/original_comm/original_comm','r') as f:
                total_count = 0
                for line in f:
                    total_count += 1
                if not timelist_original[-1] == total_count:
                    timelist_original.append(total_count)

            # update the cumulative monthly counts
            with open(self.path + "/counts/RC_Count_List", "w") as f:
                for interval in timelist_original:
                    print(str(int(interval)), end="\n", file=f)

            # update the auto-labels file and check that all negative comments
            # are removed from the dataset
            count_auto_labels = 0
            sum_auto_labels = 0
            with open(self.path + "/original_comm/auto_labels","r") as labels:
                for line in labels:
                    if line.strip() != "":
                        count_auto_labels += 1
                        sum_auto_labels += int(line.strip())
            assert count_auto_labels == sum_auto_labels

            # Measure, report and record filtering time for the entire dataset
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            with open(self.path+"rel_clean_cert.txt","w") as cert:
                cert.write("Finished relevance-filtering in {:0>2}:{:0>2}:{:05.2f} (hours,minutes,seconds)".format(int(hours),int(minutes),seconds))
                print("Finished relevance-filtering in {:0>2}:{:0>2}:{:05.2f} (hours,minutes,seconds)".format(int(hours),int(minutes),seconds))


    ## Evaluates accuracy, f1, precision and recall for the relevance classifier
    # based on the random sample from Neural_Relevance_Screen
    def eval_relevance(self,data_path=path+"/original_comm/"):

        # check that the random sample is available
        if not Path(data_path+"sample_auto_labeled.csv").is_file:
            raise Exception("Random sample of classifier output not found.")

        else: # if it is
            with open(data_path+"sample_auto_labeled.csv","r") as csvfile:
                reader = csv.reader(csvfile)

                labels = [] # container for human labels
                preds = [] # container for model predictions
                for idx,row in enumerate(reader):
                    if idx != 0:
                        if row[4].strip() == "": # check for human labels
                            raise Exception("Random sample is not fully hand-annotated")
                        else:
                            preds.append(int(row[3].strip()))
                            labels.append(int(row[4].strip()))

        label_measures = dict() # dictionary for confusion matrix

        accuracy = 0 # counter for accuracy

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
        with open(data_path+"eval_results.txt","a+") as f:

            # Record the confusion matrix
            print("Confusion matrix: "+ str(label_measures) + "\n")

            # Record the accuracy
            accuracy = float(accuracy) / float(len(labels))
            print("accuracy: " + str(accuracy))
            f.write("accuracy: " + str(accuracy) + "\n")

            if tp != 0 or fp != 0: # record the precision
                precision = float(tp) / (float(tp)+float(fp))
                print("precision: "+ str(precision))
                f.write("precision: "+ str(precision) + "\n")

            if tp != 0 or fn != 0: # record the recall
                recall = float(tp) / (float(tp)+float(fn))
                print("recall: "+ str(recall))
                f.write("recalls: "+ str(recall) + "\n")

            if tp != 0 or (fn != 0 and fp != 0): # record the f1 score
                f1 = 2 * (precision*recall)/(precision + recall)
                print("f1: "+str(f1))
                f.write("f1s: "+ str(f1) + "\n")

    ## Determines what percentage of the posts in each year was relevant based
    # on content filters

    # NOTE: Requires total comment counts (RC_Count_Total) from parser or disk

    # NOTE: Requires monthly relevant counts from parser or disk

    def Rel_Counter(self):
        if not Path(self.path + "counts/RC_Count_List").is_file():
            raise Exception(
                'Cumulative monthly comment counts could not be found')
        if not Path(self.path + "total_count/total_count").is_file():
            raise Exception(
                'Total monthly counts could not be found')

        # load the total monthly counts into a list
        monthly_list = []
        with open(self.path + "total_count/total_count", 'r') as f:
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

        relevant_year, _ = Get_Counts(self.path, frequency="yearly")

        relevant = {}
        for idx, year in enumerate(relevant_year):
            relevant[str(self.dates[0][0] + idx)] = year

        # calculate the percentage of comments in each year that was relevant and write it to file
        perc_rel = {}
        rel = open(self.path + "/perc_rel", 'a+')
        for key in relevant:
            perc_rel[key] = float(relevant[key]) / float(total_year[key])
        print(sorted(perc_rel.items()), file=rel)
        rel.close

    ## Load, calculate or re-calculate the percentage of relevant comments/year
    def Perc_Rel_RC_Comment(self):
        if Path(self.path + "/perc_rel").is_file():  # look for extant record
            # if it exists, ask if it should be overwritten
            Q = input(
                "Yearly relevant percentages are already available. Do you wish to delete them and count again [Y/N]?")

            if Q == 'Y' or Q == 'y':  # if yes
                os.remove(path + "/perc_rel")  # delete previous record
                self.Rel_Counter()  # calculate again
            else:  # if no
                print("Operation aborted")  # pass

        else:  # if there is not previous record
            self.Rel_Counter()  # calculate

    # Helper functions for select_random_comments
    def get_comment_lengths(self):
        fin = self.path + 'lda_prep/lda_prep'
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
        fns["indices_random"] = "{}/random_indices".format(self.path)
        fns["counts_random"] = "{}/Random_Count_List".format(self.path)
        fns["timedict_random"] = "{}/Random_Count_Dict".format(self.path)
        fout = fns["indices_random"]

        if (not overwrite and os.path.exists(fout)):
            print("{} exists. Skipping. Set overwrite to True to overwrite.".format(fout))
            return

        years_to_sample = sorted(years_to_sample)
        ct_peryear, ct_cumyear = Get_Counts(path=self.path, frequency="yearly")
        ct_permonth, ct_cummonth = Get_Counts(
            path=self.path, frequency="monthly")
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
