from simpletransformers.classification import ClassificationModel
from sklearn.utils import shuffle

# Class to save the training set, eval set, and create the model from simpletransformers
class Model(object):
    def __init__(self, train_df, eval_df):
        self.train_df = train_df
        self.eval_df = eval_df
        self.model = ClassificationModel('bert', 'bert-base-cased', num_labels=3, use_cuda=False,
                                         args={'fp16': False, 'num_train_epochs': 2, 'manual_seed':1,
                                               "eval_batch_size": 8,
                                               "train_batch_size": 8})

    def train(self):
        shuffled = shuffle(self.train_df)
        self.model.train_model(shuffled)

    def eval(self):
        return self.model.eval_model(self.eval_df)

    def predict(self, new_post):
        predictions, raw_outputs = self.model.predict(new_post, multi_label=True)
        return predictions
