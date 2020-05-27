from simpletransformers.classification import ClassificationModel


# Class to save the training set, eval set, and create the model from simpletransformers
class Model(object):
    def __init__(self, train_df, eval_df):
        self.train_df = train_df
        self.eval_df = eval_df
        self.model = ClassificationModel('bert', 'bert-base-uncased', num_labels=3, use_cuda=False,
                                         args={'learning_rate': 1e-5, 'num_train_epochs': 5,
                                               'overwrite_output_dir': True})

    def train(self):
        self.model.train_model(self.train_df)

    def eval(self):
        return self.model.eval_model(self.eval_df)

    def predict(self, new_post):
        predictions, raw_outputs = self.model.predict(new_post, multi_label=True)
        return predictions