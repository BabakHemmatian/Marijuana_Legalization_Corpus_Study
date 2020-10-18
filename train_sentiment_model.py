from sentiment_pretraining_defaults import *
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel

# Prepare data frames for training
data = get_comment_sentiment_df()[[TEXT_COL, LABEL_COL]]
train, _ = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

# Create model
to_train_model = ClassificationModel('roberta', MODEL_TO_TRAIN, num_labels=NUM_LABELS, use_cuda=USE_CUDA,
                                     args={'fp16': False, 'num_train_epochs': EPOCHS, 'manual_seed': MANUAL_SEED,
                                  'save_steps': SAVE_STEPS_FREQ, 'save_optimizer_and_scheduler': True,
                                  "eval_batch_size": EVAL_BATCH_SIZE,
                                  "train_batch_size": TRAIN_BATCH_SIZE})

# Train the model
to_train_model.train_model(train.iloc[STARTING_COMMENT:ENDING_COMMENT, :], output_dir=OUTPUT_DIR)

