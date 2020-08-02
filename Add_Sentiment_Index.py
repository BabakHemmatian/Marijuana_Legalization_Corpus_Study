import pathlib
import csv
import pandas as pd
from defaults import annotator_data_path

contents = pathlib.Path(annotator_data_path).iterdir()
with open("original_comm/original_comm") as f:
    original_comm = f.readlines()
with open("avg_textblob_vader/tv_sentiments") as sents:
    avg_sentiments = sents.readlines()
print(len(avg_sentiments))

for path in contents:
    df = pd.read_csv(path)
    df["Original Comment Index"] = ""
    df["Average Sentiment"] = ""
    df.to_csv("sample.csv", index=False)
    for i, row in df.iterrows():
        comment = row.text
        index = original_comm.index(comment)
        df.at[i, 'Original Comment Index'] = index
        df.at[i, 'Average Sentiment'] = avg_sentiments[index]
    df.to_csv(path)
