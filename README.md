# Reddit Discourse about Marijuana Legalization
This study, in collaboration with [Sloman Lab](https://sites.google.com/site/slomanlab/) and [AI Lab](https://brown.edu/Research/AI/people/carsten.html) at Brown University, involves applying unsupervised and supervised machine learning methods to examine temporal trends in discourse about marijuana legalization on Reddit since 2008.

Raw and processed versions of corpus and models, as well as detailed descriptions of the procedures used in this study can be found [here](https://drive.google.com/open?id=17PjV5gPub15kSaHpw9JVP1SNpj1k3vK-).

Acknowledgment: The basis for the code in the current repository is [this](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study]) study on same-sex marriage discourse, which was developed in collaboration with Sabina J. Sloman from Carnegie Mellon University, and Steven A. Sloman and Uriel Cohen Priva from Brown University.


## To setup conda environment
1. Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.
2. Clone this respository and be sure to switch to the debugging branch. 
**NOTE:** The debugging branch will soon be merged with master.
3. You will have to install spacy en_core_web_sm manually. To do this first run <kbd>pip install spacy</kbd>. Once it's installed, run <kbd>python -m spacy download en</kbd>
4. You can now create an environment using the `Marijuana_Study.yml` file. In the terminal go to the directory where you cloned this repository. Now run <kbd>conda env create -f Marijuana_Study.yml</kbd>.
5. Once the environment is created, you will need to activate the environment. To do this run <kbd>conda activate Marijuana_study</kbd>
6. You might also need nltk to run the files. To install nltk, follow [this guide](https://www.nltk.org/data.html).


## The Dataset
Our [public google drive](https://drive.google.com/drive/u/0/folders/1yx2lmbrbHr0uAA8zLj-TbHaXqOrcNhw6) contains a database file, 'reddit_50.db', 
which has the most up-to-date dataset and information used by the model. Each row corresponds to one comment, of which there are 3059959. The schema of the database is as follows:
- Comments table: 
    - original_comm (Text) : The original comments for the data set, which were extracted and parsed from pushshift's dataset of all Reddit comments [here](https://files.pushshift.io/reddit/comments/). Our data is a subset of theirs in terms of both breadth and depth.
    - original_indices (Integer): It is the index for that comment in the context of the particular month's pushshift data dump. 
    - subreddit (Text): The subreddit the comment was posted in.
    - month (Integer): The month the comment was posted.
    - year (Integer): The year the comment was posted. 
    - t_sentiments (Text): The sentiment value for each sentence in original_comm as a comma-separated string, extracted from the [TextBlob package](https://textblob.readthedocs.io/en/dev/#).
    - v_sentiments (Text): The sentiment value for each sentence in original_comm as a comma-separated string, extracted from the [Vader package](https://pypi.org/project/vaderSentiment/)
    - sentiments (Real): The sentiment taken from t_sentiments and v_sentiments averaged across all the sentences in the comment. 
    - attitude (Text): A number from 0-5 indicating the attitude of a post (0 if no attitude, 5 if a lot of attitude). Will be a comma-separated string of values if multiple human raters disagreed on the attitude.
    - persuasion (Text): A number from 0-5 indicating the degree of persuasiveness of a a post (0 if not very persuasive, 5 if very persuasve). Will be a comma-separated string of values if multiple human raters disagreed on the attitude.
    - votes (Integer): The number of votes (net of upvotes and downvotes) of a pos
    - author (Text): The comment's reddit username.
    - training (Integer): 1 if the comment will be used for training, 0 otherwise
    - topic_0...topic_49 (Real): For each topic_i column, the value represents the percentage this comment contributed to topic_i, which was extracted using the LDA model. If it did not contribute to topic_i, this value will be null. 
    - attitude_confidence (REAL): Softmaxed activation from the classifier for inferred_attitude.
    - persuasion_confidence (REAL): Softmaxed activation from the classifier for inferred_persuasion.
    - inferred_attitude (Integer): Attitude rating inferred by the classifier.
    - inferred_attitude_weight (REAL): Raw activation from the classifier for inferred_attitude.
    - inferred_persuasion (Integer): Persuasion rating inferred by the classifier.
    - inferred_persuasion_weight (REAL): Raw activation from the classifier for inferred_persuasion.

- classified_comments table: 
    - doc_id (Integer): The ID of the comment in the Comments table (related to the ROWID)
    - clauses (Text) : The clauses in the comment, separated by newline (\n)
    - genericity_pred (Text): Genericity prediction (0 is Generic, 1 is Specific, 2 in unsure)
    - genericity_softmax (Text): Genericity softmax
    - aspect_pred (Text): Aspect prediction (0 is Dynamic, 1 is Stative, 2 is unsure)
    - aspect_softmax (Text): Aspect softmax
    - boundedness_pred (Text): Boundedness prediction (0 is Bounded, 1 is Unbounded)
    - boundedness_softmax (Text): Boundedness Softmax
    - ne_tags (Text): The named entity tags obtained from Spacy for the comment.

