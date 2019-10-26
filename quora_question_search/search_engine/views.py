import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from collections import defaultdict
from django.db import connection
from django.shortcuts import render
from scipy.sparse import csr_matrix
from warnings import filterwarnings
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from elmo_helpers import load_elmo_embeddings, get_elmo_vectors
from blim import Batcher, BidirectionalLanguageModel, weight_layers
from keras.models import Model
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer, load_locabulary, get_checkpoint_paths

filterwarnings("ignore")
tf.disable_v2_behavior()
tf.reset_default_graph()

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


def build_db():
    df = pd.read_csv(os.path.join("..", "quora_question_pairs_rus.csv"))
    corpus = list(set(df["question1"])) + list(set(df["question2"]))
    conn = sqlite3.connect("quora_question_pairs_rus.db")
    db = conn.cursor()
    db.execute("""
              CREATE TABLE text_corpora
              (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              text TEXT NOT NULL,
              text_lemmatized TEXT NOT NULL);
              """)
    for text in tqdm(corpus):
        db.execute("""
                   INSERT INTO text_corpora
                   (text, text_lemmatized)
                   VALUES (?, ?);
                   """, (text, " ".join(lemmatize(text))))
        conn.commit()
    conn.close()
    return


if not os.path.isfile("quora_question_pairs_rus.db"):
    build_db()


class SearchEngine():
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        conn = connection
        db = conn.cursor()
        db.execute("""
                   SELECT text_lemmatized FROM text_corpora LIMIT 100000
                   """)
        data = db.fetchall()
        conn.close()
        return [text[0] for text in data]

    @staticmethod
    def load_matrix(filename):
        return np.load(filename)

    def load_model(self):
        pass

    def fit(self):
        pass

    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        pass

    def search(self, query):
        pass


def main(request):
    query = request.GET["query"]
    model = request.GET["model"]
    results = []
    failed_search = False
    if query and model:
        if model == "tf_idf":
            engine = TfIdfSearch()
        elif model == "bm25":
            engine = BM25Search()
        elif model == "word2vec":
            engine = Word2VecSearch()
        elif model == "fasttext":
            engine = FastTextSearch()
        elif model == "elmo":
            engine = ELMOSearch()
        elif model == "bert":
            engine = RuBERTSearch()
        results = engine.search(query)
        if results is None:
            failed_search = True
    return render(request, "main.html", {"results": results,
                  "failed_search": failed_search})
