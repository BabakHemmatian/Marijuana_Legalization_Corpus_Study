import bz2
import lzma
import zstandard as zstd
from collections import defaultdict, OrderedDict
import datetime
import hashlib
import html
import json
import multiprocessing
import spacy
import nltk
import numpy as np
import os
from pathlib2 import Path
import pickle
import re
import time
import subprocess
import sys
from textblob import TextBlob
from config import *
from Utils import *

## This needs to be importable from the main module for multiprocessing
# https://stackoverflow.com/questions/24728084/why-does-this-implementation-of-multiprocessing-pool-not-work

def parse_one_month_wrapper(args):
    year, month, on_file, kwargs = args
    Parser(**kwargs).parse_one_month(year, month, on_file)

# Create global helper function for formatting names of data files
# Format dates to be consistent with pushshift file names
def format_date(yr, mo):
    if len(str(mo)) < 2:
        mo = '0{}'.format(mo)
    assert len(str(yr)) == 4
    assert len(str(mo)) == 2
    return "{}-{}".format(yr, mo)

# Raw Reddit data filename format. The compression types for dates handcoded
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

# based on provided dates, gather a list of months for which data is already
# available
on_file = []
for date in dates:
    mo,yr = date[0],date[1]
    proper_filename = get_rc_filename(mo,yr)
    if Path(path+'/'+proper_filename).is_file():
        on_file.append(proper_filename)

## Define the parser class

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
    def __init__(self, clean_raw=CLEAN_RAW, dates=dates,
                 download_raw=DOWNLOAD_RAW, hashsums=None, NN=NN, path=path,
                 legality=legality, marijuana=marijuana, stop=stop,
                 write_original=WRITE_ORIGINAL, vote_counting=vote_counting, author=author, sentiment=sentiment, on_file=on_file):
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

    # Download Reddit comment data
    def download(self, year=None, month=None, filename=None):
        assert not all([isinstance(year, type(None)),
                         isinstance(month, type(None)),
                         isinstance(filename, type(None))
                         ])
        assert isinstance(filename, type(None)) or (isinstance(year, type(None))
                                                    and isinstance(month, type(None)))
        BASE_URL = 'https://files.pushshift.io/reddit/comments/'
        if not isinstance(filename, type(None)):
            url = BASE_URL+filename
        else:
            url = BASE_URL+get_rc_filename(year, month)
        print('Sending request to {}.'.format(url))
        os.system('cd {} && wget {}'.format(self.path, url))

    # # Get Reddit compressed data file hashsums to check downloaded files' integrity
    # def Get_Hashsums(self):
    #     # notify the user
    #     print('Retrieving hashsums to check file integrity')
    #     # set the URL to download hashsums from
    #     url = 'https://files.pushshift.io/reddit/comments/sha256sum.txt'
    #     # remove any old hashsum file
    #     if Path(self.path+'/sha256sum.txt').is_file():
    #         os.remove(self.path+'/sha256sum.txt')
    #     # download hashsums
    #     os.system('cd {} && wget {}'.format(self.path, url))
    #     # retrieve the correct hashsums
    #     hashsums = {}
    #     with open(self.path+'/sha256sum.txt', 'rb') as f:
    #         for line in f:
    #             line = line.decode("utf-8")
    #             if line.strip() != "" and ".xz" not in line:
    #                 (val, key) = str(line).split()
    #                 hashsums[key] = val
    #     return hashsums
    #
    # # calculate hashsums for downloaded files in chunks of size 4096B
    # def sha256(self, fname):
    #     hash_sha256 = hashlib.sha256()
    #     with open("{}/{}".format(self.path, fname), "rb") as f:
    #         for chunk in iter(lambda: f.read(4096), b""):
    #             hash_sha256.update(chunk)
    #     return hash_sha256.hexdigest()

    def _clean(self, text):

        # check input arguments for valid type
        assert type(text) is str

        # remove stopwords --> check to see if apostrophes are properly encoded

        stop_free = " ".join([i for i in text.lower().split() if i.lower() not
                              in self.stop])

        replace = {"should've": "should", "mustn't": "mustn",
                   "shouldn't": "shouldn", "couldn't": "couldn", "shan't": "shan",
                   "needn't": "needn", "-": ""}
        substrs = sorted(replace, key=len, reverse=True)
        regexp = re.compile('|'.join(map(re.escape, substrs)))
        stop_free = regexp.sub(
            lambda match: replace[match.group(0)], stop_free)

        # remove special characters
        special_free = ""
        for word in stop_free.split():
            if "http" not in word and "www" not in word:  # remove links
                word = re.sub('[^A-Za-z0-9]+', ' ', word)
                if word.strip() != "":
                    special_free = special_free+" "+word.strip()

        # check for stopwords again
        special_free = " ".join([i for i in special_free.split() if i not in
                                 self.stop])

        return special_free

    # define the preprocessing function to add padding and remove punctuation, special characters and stopwords (neural network)
    def NN_clean(self, text):

        # check input arguments for valid type
        assert type(text) is list or type(text) is str

        # create a container for preprocessed sentences
        cleaned = []

        for index, sent in enumerate(text):  # iterate over the sentences
            special_free = self._clean(sent)

            # add sentence and end of comment padding
            if special_free.strip() != "":
                padded = special_free+" *STOP*"
                if index+1 == len(text):
                    padded = padded+" *STOP2*"
                cleaned.append(padded)
            elif special_free.strip() == "" and len(text) != 1 and len(cleaned) != 0 and index+1 == len(text):
                cleaned[-1] = cleaned[-1]+" *STOP2*"

        return cleaned

    # define the preprocessing function to lemmatize, and remove punctuation,
    # special characters and stopwords (LDA)

    # NOTE: Since LDA doesn't care about sentence structure, unlike NN_clean,
    # the entire comment should be fed into this function as a continuous string

    # NOTE: The Reddit dataset seems to encode the quote blocks as just new
    # lines. Therefore, there is no way to get rid of quotes

    def LDA_clean(self, text):

        special_free = self._clean(text)

        # load lemmatizer with automatic POS tagging
        lemmatizer = spacy.load('en', disable=['parser', 'ner'])
        # Extract the lemma for each token and join
        lemmatized = lemmatizer(special_free)
        normalized = " ".join([token.lemma_ for token in lemmatized])

        return normalized

    def get_parser_fns(self, year=None, month=None):
        assert ((isinstance(year, type(None)) and isinstance(month, type(None))) or
                 (not isinstance(year, type(None)) and not isinstance(month, type(None))))
        if isinstance(year, type(None)) and isinstance(month, type(None)):
            suffix = ""
        else:
            suffix = "-{}-{}".format(year, month)
        fns = dict((("lda_prep", "{}/lda_prep{}".format(self.path, suffix)),
                  ("original_comm", "{}/original_comm{}".format(self.path, suffix)),
                  ("original_indices", "{}/original_indices{}".format(self.path, suffix)),
                  ("counts", "{}/RC_Count_List{}".format(self.path, suffix)),
                  ("timedict", "{}/RC_Count_Dict{}".format(self.path, suffix)),
                  ("indices_random", "{}/random_indices".format(self.path)),
                  ("counts_random", "{}/Random_Count_List".format(self.path)),
                  ("timedict_random", "{}/Random_Count_Dict".format(self.path))
                  ))
        if self.NN:
            fns["nn_prep"] = "{}/nn_prep{}".format(self.path, suffix)
        if self.vote_counting:
            fns["votes"] = "{}/votes{}".format(self.path, suffix)
        if self.author:
            fns["author"] = "{}/author{}".format(self.path, suffix)
        if self.sentiment:
            fns["sentiments"] = "{}/sentiments{}".format(self.path, suffix)
        return fns

    # NOTE: Parses for LDA if NN = False
    # NOTE: Saves the text of the non-processed comment to file as well if write_original = True
    def parse_one_month(self, year, month, on_file):
        timedict = dict()

        if self.NN:  # if parsing for a neural network
            # import the pre-trained PUNKT tokenizer for determining sentence boundaries
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        decoder = json.JSONDecoder()

        # prepare files
        # get the relevant compressed data file name
        filename = get_rc_filename(year, month)
        # if the file is available on disk and download is on, prevent deletion
        if not filename in self.on_file and self.download_raw:
            self.download(year, month)  # download the relevant file

            # BUG: Hashsum check is currently non-functional, as the sums for
            # months later than 11-2017 have not been calculated. Until a
            # future fix, the following code section should remain commented

            # check data file integrity and download again if needed
            # calculate hashsum for the data file on disk
            # filesum = self.sha256(filename)
            # attempt = 0  # number of hashsum check trials for the current file
            # # if the file hashsum does not match the correct hashsum
            # while filesum != self.hashsums[filename]:
            #     attempt += 1  # update hashsum check counter
            #     if attempt == 5:  # if failed hashsum check three times, ignore the error to prevent an infinite loop
            #         print("Failed to pass hashsum check 5 times. Ignoring the error.")
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

        # Get names of processing files
        fns = self.get_parser_fns(year, month)

        # create a file to write the processed text to
        if self.NN:  # if doing NN
            fout = open(fns["nn_prep"], 'w')
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
            sentiments = open(fns["sentiments"],'w')

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
            fin = lzma.open(filename,'r')
        elif '.bz2' in filename:
            fin = bz2.BZ2File(self.path+'/'+filename, 'r')
        else:
            raise Exception('File format not recognized')

        # read data
        for line in fin:  # for each comment
            main_counter += 1  # update the general counter

            # decode and parse the json, and turn it into regular text
            if not '.zst' in filename:
                comment = line.decode()
            comment = decoder.decode(comment)
            original_body = html.unescape(comment["body"])  # original text

            # filter comments by relevance to the topic according to regex
            if any(not exp.search(original_body) is None for exp in marijuana) and any(not exp.search(original_body) is None for exp in legality):

                # preprocess the comments
                if self.NN:

                    # Tokenize the sentences
                    body = sent_detector.tokenize(
                        original_body)
                    body = self.NN_clean(body)  # clean the text for NN
                    if len(body) == 0:  # if the comment body is not empty after preprocessing
                        continue

                    # If calculating sentiment, write the average sentiment to
                    # file. Range is -1 to 1, with values below 0 meaning neg
                    # sentiment
                    if self.sentiment:
                        blob = TextBlob(original_body)
                        print(blob.sentiment[0],file=sentiments)

                    # if we want to write the original comment to disk
                    if self.write_original:
                        original_body = original_body.replace(
                            "\n", "")  # remove mid-comment lines
                        # record the original comment
                        print(" ".join(original_body.split()), file=foriginal)
                        # record the main index
                        print(main_counter, file=main_indices)

                    for sen in body:  # for each sentence in the comment
                        # remove mid-comment lines
                        sen = sen.replace("\n", "")

                        # print the processed sentence to file
                        print(" ".join(sen.split()), end=" ", file=fout)

                    # ensure that each comment is on a separate line
                    print("\n", end="", file=fout)

                else:  # if doing LDA

                    # clean the text for LDA
                    body = self.LDA_clean(original_body)

                    if not body.strip():  # if the comment is not empty after preprocessing
                        continue

                    # If calculating sentiment, write the average sentiment to
                    # file. Range is -1 to 1, with values below 0 meaning neg
                    # sentiment
                    if self.sentiment:
                        blob = TextBlob(original_body)
                        print(blob.sentiment[0],file=sentiments)

                    # if we want to write the original comment to disk
                    if self.write_original:
                        original_body = original_body.replace(
                            "\n", "")  # remove mid-comment lines
                        # record the original comment
                        print(" ".join(original_body.split()), file=foriginal)
                        # record the index in the original files
                        print(main_counter, file=main_indices)

                    # remove mid-comment lines
                    body = body.replace("\n", "")
                    body = " ".join(body.split())

                    # print the comment to file
                    print(body, sep=" ", end="\n", file=fout)

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

        # write the total number of posts from the month to disk to aid in
        # calculating proportion relevant if calculate_perc_rel = True
        if calculate_perc_rel:
            with open(self.path+"/total_count",'a+') as counter_file:
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
        ccount.close()
        with open(fns["timedict"], "wb") as wfh:
            pickle.dump(timedict, wfh)

        # timer
        print("Finished parsing "+filename+" at " + time.strftime('%l:%M%p'))

        # TODO: Fix clean_raw so that it won't delete
        # if the user wishes compressed data files to be removed after processing
        if self.clean_raw and filename not in self.on_file:
            print("Cleaning up {}/{}.".format(self.path, filename))
            # delete the recently processed file
            os.system('cd {} && rm {}'.format(self.path, filename))

        return

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
                            print(month+" "+str(docs), end='\n', file=tdfh)
                            # Use timedict data to populate counts file
                            cumul_docs += docs
                            print(cumul_docs, end='\n', file=cfh)
                continue
            subprocess.call("cat "+" ".join(fns_)+"> "+fns[kind], shell=True)

    def parse(self):
        # get the correct hashsums to check file integrity
        # self.hashsums = self.Get_Hashsums()

        # Parallelize parsing by month
        # NOTE: For best results, set the number of processes in the following
        # line based on (number of physical cores)*(hyper-threading multiplier)
        # -1 (for synchronization overhead)
        pool = multiprocessing.Pool(processes=7)
        inputs = [(year, month, self.on_file, self.__dict__) for year, month in self.dates]
        pool.map(parse_one_month_wrapper, inputs)

        # timer
        print("Finished parsing at " + time.strftime('%l:%M%p'))

        # Pool parsing data from all files
        self.pool_parsing_data()

    # Function to call parser when needed and parse comments
    # TODO: Replace mentions of Vote in this file with mentions of sample_ratings
    # TODO: Add main counter and original comments and indices to this function
    def Parse_Rel_RC_Comments(self):
        # if preprocessed comments are available, ask if they should be rewritten
        if (self.NN and Path(self.path+"/nn_prep").is_file()) or (not self.NN and Path(self.path+"/lda_prep").is_file()):
            Q = input(
                "Preprocessed comments are already available. Do you wish to delete them and parse again [Y/N]?")
            if Q == 'Y' or Q == 'y':  # if the user wishes to overwrite the comments
                # delete previous preprocessed data
                if self.NN:  # for NN
                    os.remove(self.path+"/nn_prep")
                elif not self.NN:  # for LDA
                    os.remove(self.path+"/lda_prep")
                if Path(self.path+"/original_indices").is_file() and self.write_original:
                    os.remove(self.path+"/original_indices")
                if Path(self.path+"/original_comm").is_file() and self.write_original:
                    os.remove(self.path+"/original_comm")
                if Path(self.path+"/votes").is_file() and self.vote_counting:
                    os.remove(self.path+"/votes")
                if Path(self.path+"/author").is_file() and self.author:
                    os.remove(self.path+"/author")
                if Path(self.path+"/sentiments").is_file() and self.sentiment:
                    os.remove(self.path+"/sentiments")
                if Path(self.path+"/RC_Count_List").is_file():
                    os.remove(self.path+"/RC_Count_List")
                if Path(self.path+"/RC_Count_Dict").is_file():
                    os.remove(self.path+"/RC_Count_Dict")

                # timer
                print("Started parsing at " + time.strftime('%l:%M%p'))
                self.parse()

            else:  # if preprocessed comments are available and
                # the user does not wish to overwrite them
                print("Checking for missing files")

                # check for other required files aside from main data
                missing_files = 0

                if not Path(self.path+"/RC_Count_List").is_file():
                    missing_files += 1

                if not Path(self.path+"/votes").is_file() and self.vote_counting:
                    missing_files += 1

                if not Path(self.path+"/author").is_file() and self.author:
                    missing_files += 1

                if not Path(self.path+"/sentiments").is_file() and self.sentiment:
                    missing_files += 1

                # if there are missing files, delete any partial record and parse again
                if missing_files != 0:
                    print("Deleting partial record and parsing again")

                    if Path(self.path+"/votes").is_file():
                        os.remove(self.path+"/votes")

                    if Path(self.path+"/author").is_file():
                        os.remove(self.path+"/author")

                    if Path(self.path+"/sentiments").is_file():
                        os.remove(self.path+"/sentiments")

                    if self.NN:  # for NN
                        os.remove(self.path+"/nn_prep")

                    elif not self.NN:  # for LDA
                        os.remove(self.path+"/lda_prep")

                    if Path(self.path+"/RC_Count_List").is_file():
                        os.remove(self.path+"/RC_Count_List")

                    if Path(self.path+"/RC_Count_Dict").is_file():
                        os.remove(self.path+"/RC_Count_Dict")

                    # timer
                    print("Started parsing at " + time.strftime('%l:%M%p'))
                    self.parse()

        else:
            if Path(self.path+"/RC_Count_List").is_file():
                os.remove(self.path+"/RC_Count_List")
            if Path(self.path+"/votes").is_file() and self.vote_counting:
                os.remove(self.path+"/votes")
            if Path(self.path+"/author").is_file() and self.author:
                os.remove(self.path+"/author")
            if Path(self.path+"/sentiments").is_file() and self.sentiment:
                os.remove(self.path+"/sentiments")
            if Path(self.path+"/original_comm").is_file() and self.write_original:
                os.remove(self.path+"/original_comm")
            if Path(self.path+"/original_indices").is_file() and self.write_original:
                os.remove(self.path+"/original_indices")

            # timer
            print("Started parsing at " + time.strftime('%l:%M%p'))
            self.parse()

    # determine what percentage of the posts in each year was relevant based on content filters
    # NOTE: Requires total comment counts (RC_Count_Total) from
    # http://files.pushshift.io/reddit/comments/, not available after 2-2018
    # TODO: Add support based on (total_count) for periods after 2-2018
    # NOTE: Requires monthly relevant counts from parser or disk
    def Rel_Counter(self):
        if not Path(self.path+"/RC_Count_List").is_file():
            raise Exception(
                'Cumulative monthly comment counts could not be found')
        if not Path(self.path+"/total_count").is_file():
            raise Exception(
                'Total monthly counts could not be found')

        # load the total monthly counts into a list
        monthly_list = []
        with open(self.path+"/total_count",'r') as f:
            for line in f:
                line = line.replace("\n", "")
                if line.strip() != "":
                    monthly_list.append(line)

        d = {}
        for idx,tuples in enumerate(self.dates):
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
            relevant[str(self.dates[0][0]+idx)] = year

        # calculate the percentage of comments in each year that was relevant and write it to file
        perc_rel = {}
        rel = open(self.path+"/perc_rel", 'a+')
        for key in relevant:
            perc_rel[key] = float(relevant[key]) / float(total_year[key])
        print(sorted(perc_rel.items()), file=rel)
        rel.close

    # Load, calculate or re-calculate the percentage of relevant comments/year
    def Perc_Rel_RC_Comment(self):
        if Path(self.path+"/perc_rel").is_file():  # look for extant record
            # if it exists, ask if it should be overwritten
            Q = input(
                "Yearly relevant percentages are already available. Do you wish to delete them and count again [Y/N]?")

            if Q == 'Y' or Q == 'y':  # if yes
                os.remove(path+"/perc_rel")  # delete previous record
                self.Rel_Counter()  # calculate again
            else:  # if no
                print("Operation aborted")  # pass

        else:  # if there is not previous record
            self.Rel_Counter()  # calculate

    # Helper functions for select_random_comments
    def get_comment_lengths(self):
        fin = self.path+'/lda_prep'
        with open(fin, 'r') as fh:
            return [len(line.split()) for line in fh.read().split("\n")]

    def _select_n(self, n, iterable):
        if len(iterable) < n:
            return iterable
        return np.random.choice(iterable, size=n, replace=False)

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
        assert all([early_years[i] == early_years[i-1]+1 for i in range(1,
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
                if fyear-1 in ct_lu_by_year:
                    start = ct_cumyear[ct_lu_by_year[fyear-1]]
                else:
                    start = 0
                end = ct_cumyear[ct_lu_by_year[lyear]]
                ixs_longenough =[ ix for ix in range(start, end) if lens[ix] >=
                                 min_comm_length]
                ixs = sorted(self._select_n(n, ixs_longenough))
                for ix in ixs:
                    nixs[self.dates[[ ct > ix for ct in ct_cummonth ].index(True)]] +=1
                assert sum(nixs.values()) == len(ixs)
                wfh.write('\n'.join(map(str, ixs)))
                wfh.write('\n')
            for year in later_years:
                start = ct_cumyear[ct_lu_by_year[year-1]]
                end = ct_cumyear[ct_lu_by_year[year]]
                ixs_longenough =[ ix for ix in range(start, end) if lens[ix] >=
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
                    print(month+" "+str(docs), end='\n', file=tdfh)
                    # Use timedict data to populate counts file
                    cumul_docs += docs
                    print(cumul_docs, end='\n', file=cfh)
