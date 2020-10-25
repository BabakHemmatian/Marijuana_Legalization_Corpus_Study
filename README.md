# Reddit Discourse about Marijuana Legalization
This study, in collaboration with [Sloman Lab](https://sites.google.com/site/slomanlab/) and [AI Lab](https://brown.edu/Research/AI/people/carsten.html) at Brown University, involves applying unsupervised and supervised machine learning methods to examine temporal trends in discourse about marijuana legalization on Reddit since 2008.

Raw and processed versions of corpus and models, as well as detailed descriptions of the procedures used in this study can be found [here](https://drive.google.com/open?id=17PjV5gPub15kSaHpw9JVP1SNpj1k3vK-).

Acknowledgment: The basis for the code in the current repository is [this](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study]) study on same-sex marriage discourse, which was developed in collaboration with Sabina J. Sloman from Carnegie Mellon University, and Steven A. Sloman and Uriel Cohen Priva from Brown University.



## The Dataset

Our [public google drive](https://drive.google.com/drive/u/0/folders/1yx2lmbrbHr0uAA8zLj-TbHaXqOrcNhw6) contains a database file, 'reddit_50.db', 
which has the most up-to-date dataset and information used by the model. Each row corresponds to one comment, of which there are 3059959. The schema of the database is as follows:
1. Comments table: 
    - original_comm (String) : The original comments for the data set, which were extracted and parsed from pushshift [Need link] and filtered to posts pertaining to Marijuana Legalization.
    - original_indices: I don't know what this means fully. 
    - subreddit (String): The subreddit the comment was posted in.
    - month (Integer): The month the comment was posted.
    - year (Integer): The year the comment was posted. 
    - average_comment (String): 
    - t_sentiments (String): The sentiment value for each sentence in original_comm as a comma-separated string, extracted from the [TextBlob package](https://textblob.readthedocs.io/en/dev/#).
    - v_sentiments: The sentiment value for each sentence in original_comm as a comma-separated string, extracted from the [Vader package](https://pypi.org/project/vaderSentiment/)
    - votes (Integer): The number of votes (net of upvotes and downvotes) of a pos
    - author (String): The comment's reddit username
    - relevance (Integer): A number from 0-3 indicating the relevance of a post (0 if irrelevant and 3 if very relevant). Will be a comma-separated string of values if multiple human raters disagreed on the relevance.
    - attitude (String): A number from 0-5 indicating the attitude of a post (0 if no attitude, 5 if a lot of attitude). Will be a comma-separated string of values if multiple human raters disagreed on the attitude.
    - persuasion (String): A number from 0-5 indicating the degree of persuasiveness of a a post (0 if not very persuasive, 5 if very persuasve). Will be a comma-separated string of values if multiple human raters disagreed on the attitude.
    - topic_0...topic_n (Double): For each topic_i column, the value represents the percentage this comment contributed to topic_i, which was extracted using the LDA model. If it did not contribute to topic_i, this value will be null. 
    - training (Integer): 1 if the comment will be used for training, 0 otherwise