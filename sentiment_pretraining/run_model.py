import pandas as pd

# Prepare data sets
from sentiment_pretraining.DataPrep import DataPrep
from sentiment_pretraining.Model import Model

prep = DataPrep()
prep.prep_data()

# Prepare data frames for training

train_df = pd.read_csv('train.csv', header=None)
eval_df = pd.read_csv('test.csv', header=None)

# Run model
model = Model(train_df, eval_df)

model.train()

# Evaluate model
result, model_outputs, wrong_predictions = model.eval()
