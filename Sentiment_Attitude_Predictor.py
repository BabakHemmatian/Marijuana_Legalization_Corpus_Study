import pathlib
import csv
import pandas as pd
from defaults import annotator_data_path

contents = pathlib.Path(annotator_data_path).iterdir()
with open("original_comm/original_comm") as f:
    original_comm = f.readlines()
with open("avg_textblob_vader/tv_sentiments") as sents:
    avg_sentiments = sents.readlines()
for path in contents:
    df = pd.read_csv(path)
    df["Original Comment Index"] = ""
    df["Average Sentiment"] = ""
    print(path)
    for i, row in df.iterrows():
        comment = row['text']
        try:
            index = original_comm.index(comment)
            df.at[i, 'Original Comment Index'] = index
            df.at[i, 'Average Sentiment'] = avg_sentiments[index]
        except ValueError:
            pass
    df.to_csv(path)
