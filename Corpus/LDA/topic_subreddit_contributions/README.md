The files in this folder are used to generate bar plots plotting the percentage contribution of topics to a particular comment extracted from the LDA model, for each of the top ten subreddits in the dataset. For a given plot,
the y-axis is the percentage contribution to a particular topic, and the x-axis corresponds to bars for each of the top ten subreddits.

First run `topic_subreddit_contributions_extraction.py`, which requires `reddit_50_both_inferred.db` and `data_for_R-f.csv`, which
can be downloaded from the LDA folder. This generates a file called `topic_subreddit_contributions.csv`, which is required to run `topic_subreddit_visual.py`
which generates the bar plots. 
