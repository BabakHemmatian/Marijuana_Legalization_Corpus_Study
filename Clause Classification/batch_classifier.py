import argparse
import io
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--array', type = int)
    args = argparser.parse_args()

    # array is the id of the node running
    array = args.array

    segmented_file = '/users/asriva11/data/asriva11/comments_segmentation/discourse_parser/segmented_comments_new/segmented_machine_' + str(array) + '.txt'
    clausified_file = '/users/asriva11/data/asriva11/clause_classification/clausified_batches/clausified_segmented_machine_' + str(array) + '.txt'

    os.system('python /users/asriva11/data/asriva11/clause_classification/clausify_batch.py ' + segmented_file)
    os.system('python /users/asriva11/data/asriva11/clause_classification/classifier.py ' + clausified_file)
