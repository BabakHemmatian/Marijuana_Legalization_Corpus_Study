1. Clause annotation contains a file used for targeted sampling of documents for human annotation. If you do not plan to develop your novel corpus, you can safely disregard this folder.
2. Clause-level Models Development contains the TensorFlow code used for training models to classify the three focal clause-level properties (genericity, fundamental aspect and boundedness) using transformer-based neural networks.
3. Clause Classification contains the TensorFlow and SLURM code used for heavily parallelized classification of the Reddit Marijuana Legalization Corpus using the models mentioned above.
4. Analyses contains code for evaluation of the classification results, producing results that are reported in Babak Hemmatian's dissertation.
