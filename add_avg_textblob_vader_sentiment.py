from reddit_parser import Parser
import os
from defaults import model_path

theparser = Parser(machine="local")

assert (os.path.exists(model_path + "/t_sentiments/t_sentiments"))
assert (os.path.exists(model_path + "/v_sentiments/v_sentiments"))

with open(model_path + "/avg_textblob_vader/tv_sentiments", "a+") as to_write:
    for line_vader, line_text_blob in zip(open(model_path + "/t_sentiments/t_sentiments", "r"),
                                          open(model_path + "/v_sentiments/v_sentiments", "r")):
        vader_doc_array = [float(val) for val in line_vader.split(",")]
        text_blob_doc_array = [float(val) for val in line_text_blob.split(",")]
        assert(len(vader_doc_array) == len(text_blob_doc_array))
        avg_vader_doc = sum(vader_doc_array) / len(vader_doc_array)
        avg_text_blob_doc = sum(text_blob_doc_array) / len(text_blob_doc_array)
        total_avg = (avg_vader_doc + avg_text_blob_doc) / 2
        to_write.write(str(total_avg) + "\n")



