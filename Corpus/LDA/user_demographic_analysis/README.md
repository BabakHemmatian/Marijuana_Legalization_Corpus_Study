The files in this folder are used to correlate user demographic information with the relevant topics in our dataset.
 
Specifically, we correlated each of the topics with religiosity and percentage GOP for a given author's geographic location. We exclude comments whose authors 
are outside of the US or whose location was inferred based on less than 10 Reddit comments. 

First,`topic_demographic_preprocessing.py` should be run to generate 50 txt files (one for each topic). Each of these text files stores the religiosity, percentage GOP, and percentage
contribution to that particular topic for each of the comments in the database (>3M). Topic contribution is weighted by the number of words in that comment
assigned to that topic. Then, to generate the correlations, run `topic_demographic_correlate.py`.

To run the files, please contact the administrators
as it requires sensitive, user-level location info.
