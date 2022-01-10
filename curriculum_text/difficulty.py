import pandas as pd
from box import Box
from typing import *
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import re
import logging
from scipy import spatial


class DifficultyMeasure(object):
    def __init__(self, config: Box, train_df: pd.DataFrame, label_desc: List):
        self.config = config
        self.train_df = train_df
        self.train_df.to_csv("notebooks/train_df.csv")
        self.label_desc = label_desc
        self.replace_invalid_labels()

    def replace_invalid_labels(self):
        new_list = []
        replace_dict = {
            "EducationalInstitution": "Educational",
            "MeanOfTransportation": "Transportation",
            "OfficeHolder": "Office",
            "WrittenWork": "Written",
            "NaturalPlace": "Natural",
            "1 star": "one",
            "2 star": "two",
            "3 stars": "three",
            "4 stars": "four",
            "5 stars": "five",
            "Sci/Tech": "Science",
        }

        for label in self.label_desc:
            if label in replace_dict.keys():
                new_list.append(replace_dict[label])
            else:
                new_list.append(label)
        self.label_desc = new_list
        print(self.label_desc)

    def calc_length(self):
        self.train_df.loc[:, "length"] = self.train_df["text"].str.len()
        # remove empty samples
        self.train_df = self.train_df[self.train_df.length > 0]

    def vector_distance(self):
        self._load_w2v()

        # get label descriptions to map to word vectors
        label_dict = {}
        [label_dict.update({idx: value}) for idx, value in enumerate(self.label_desc)]
        self.train_df.loc[:, "label_desc"] = self.train_df["label"].map(label_dict)
        # remove nans
        # self.train_df = self.train_df[~self.train_df.isna()]

        logging.info(f"Calculating document vectors for training data...")
        self.train_df.loc[:, "doc_vec"] = self.train_df.text.apply(self._calc_doc_vec)
        logging.info(f"Fetching word vectors for labels...")
        label_vec_dict = self._get_label_vecs(self.label_desc)
        self.train_df.loc[:, "label_vec"] = self.train_df.label_desc.map(label_vec_dict)
        logging.info(f"Calculating distance measure")
        self.train_df.loc[:, "distance"] = self.train_df.apply(
            lambda x: spatial.distance.cosine(x.doc_vec, x.label_vec), axis=1
        )
        self.train_df.loc[:, "label_rank"] = self.train_df.groupby("label")[
            "distance"
        ].rank("first", ascending=True)
        return self.train_df.sort_values(by="label_rank", ascending=True)

    def _load_w2v(self):
        glove_file = "data/glove.6B.50d.txt"
        logging.info(f"Loading GloVe vectors...")
        self.wv = KeyedVectors.load_word2vec_format(
            glove_file, binary=False, no_header=True
        )
        logging.info("Done!")

    def _calc_doc_vec(self, text):
        word_vecs = []
        if type(text) == float:
            print(text)
        for token in text.split(" "):
            try:
                vec = np.array(self.wv[token])
            except:
                # skip token if OOV word
                continue
            word_vecs.append(vec)

        # take simple average of word vectors
        doc_vec = np.mean(word_vecs, axis=0)
        return doc_vec

    def _get_label_vecs(self, labels: List) -> Dict:

        label_vec_dict = {}
        for label in labels:
            label_clean = re.findall(r"[\w']+|[.,!?;]", label)[-1]
            vec = self.wv[label_clean.lower()]
            label_vec_dict.update({label: vec})
        return label_vec_dict

    def length(self):
        self.calc_length()
        self.train_df.loc[:, "label_rank"] = self.train_df.groupby("label")[
            "length"
        ].rank("first", ascending=True)
        return self.train_df.sort_values(by="label_rank", ascending=True)

    def reverse_length(self):
        self.calc_length()
        self.train_df.loc[:, "label_rank"] = self.train_df.groupby("label")[
            "length"
        ].rank("first", ascending=True)
        return self.train_df.sort_values(by="label_rank", ascending=False)
