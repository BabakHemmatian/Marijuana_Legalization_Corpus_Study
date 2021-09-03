# Clause Level Analyses

The contents of this folder were used to perform different analyses on the final reddit database containing clause classifications. 

## Files and directories in this directory - 

1. phi_values - Contains python files containing variables with phi value dictionaries for genericity, aspect, and boundedness. Contains both aggregate phis and temporal phis. The aggregate phi variables are dictionaries where each entry is of the format 'topic_number': 'total phi score'. The temporal phi variables are dictionaries where each entry is of the format 'month-year': dict of aggregate phis of that month.
2. failed_segmentation_checks.py - this file was used to generate a figure to analyze the distribution of the comments that failed to segment by subreddits and time.
3. topic_analysis.ipynb - this file was used to generate aggregate and temporal phis for genericity, aspect, and boundedness. Furthermore, it was used to generate figures for topic-generality, topic-aspect, and topic-boundedness. Further, it was also used to generate figures for temporal change in generality, aspect, and boundedness per topic. 
4. word_topic_assignments.py -