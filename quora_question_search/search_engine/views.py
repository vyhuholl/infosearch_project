import sys
import pickle
import logging
import sqlite3
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from os import path
from tqdm import tqdm
from django.db import connection
from django.shortcuts import render
from scipy.sparse import csr_matrix
from warnings import filterwarnings
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from elmo_helpers import load_elmo_embeddings, get_elmo_vectors
from blim import Batcher, BidirectionalLanguageModel, weight_layers
from keras.models import Model
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer, load_locabulary, get_checkpoint_paths
from gensim.models.keyedvectors import KeyedVectors, Word2VecKeyedVectors

filterwarnings("ignore")
tf.disable_v2_behavior()
tf.reset_default_graph()

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


def build_db():
    df = pd.read_csv(path.join("..", "quora_question_pairs_rus.csv"))
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


if not path.isfile("quora_question_pairs_rus.db"):
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
        return np.array([text[0] for text in data])

    @staticmethod
    def load_matrix(filename):
        return np.load(filename, allow_pickle=True)

    def load_model(self):
        pass

    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        pass

    def search(self, query):
        pass


class TfIdfSearch(SearchEngine):
    def __init__(self):
        super(SearchEngine, self).__init__()
        self.data = self.load_data()
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        if not path.isfile("tfidf_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("tfidf_matrix.npy")

    def fit_transform(self):
        TF = self.vectorizer.fit_transform(self.texts)
        IDF = np.array([((TF.getnnz(axis=1)).sum() - y) / y
                        for y in TF.nonzero()[0]])
        TF_IDF = np.matmul(TF.transpose(), IDF)
        self.matrix = csr_matrix((TF_IDF, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        np.save("tfidf_matrix.npy", self.matrix)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        query_vec = self.transform([query])
        result = np.array((query_vec * self.matrix).todense())[0]
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        conn = connection
        db = conn.cursor()
        for index in indices:
            text = db.execute(f"""SELECT text from text_corpora
                                 WHERE id=%d""", (index,))
            best_match.append((text, result[index]))
        db.close()
        return best_match


class BM25Search(SearchEngine):  # b=0, k=2
    def __init__(self):
        super(SearchEngine, self).__init__()
        self.data = self.load_data()
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        if not path.isfile("bm25_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("bm25_matrix.npy")

    def fit_transform(self):
        TF = self.vectorizer.fit_transform(self.texts)
        IDF = np.array([((TF.getnnz(axis=1)).sum() - y) / y
                        for y in TF.nonzero()[0]])
        BM25 = IDF * TF.data * 3 / (TF.data + 3)
        self.matrix = csr_matrix((BM25, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        np.save("bm25_matrix.npy", self.matrix)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        query_vec = self.transform([query])
        result = np.array((query_vec * self.matrix).todense())[0]
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        conn = connection
        db = conn.cursor()
        for index in indices:
            text = db.execute(f"""SELECT text from text_corpora
                                 WHERE id=%d""", (index,))
            best_match.append((text, result[index]))
        db.close()
        return best_match


class Word2VecSearch(SearchEngine):
    def __init__(self):
        super(SearchEngine, self).__init__()
        self.data = self.load_data()
        self.model = self.load_model()
        if not path.isfile("word2vec_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("word2vec_matrix.npy")

    def load_model(self):
        return Word2VecKeyedVectors.load_word2vec_format(
            path.join("..", "model_word2vec", "model.bin"), binary=True)

    def transform(self, text):
        lemmas_vectors = np.zeros((len(text), self.model.vector_size))
        text_vec = np.zeros((self.model.vector_size,))
        for index, lemma in enumerate(text):
            if lemma in self.model.vocab:
                lemmas_vectors[index] = self.model[lemma]
        if lemmas_vectors.shape[0] != 0:
            text_vec = np.mean(lemmas_vectors, axis=0)
        return text_vec

    def fit_transform(self):
        self.matrix = np.zeros((100000, self.model.vector_size))
        for i in tqdm(range(100000)):
            self.matrix[i] = self.transform(self.data[i])
        np.save("word2vec_matrix.npy", self.matrix)

    def search(self, query):
        query_vec = np.transpose(self.transform(lemmatize(query)))
        result = np.matmul(self.matrix, query_vec)
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        conn = connection
        db = conn.cursor()
        for index in indices:
            text = db.execute(f"""SELECT text from text_corpora
                                 WHERE id=%d""", (index,))
            best_match.append((text, result[index]))
        db.close()
        return best_match


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
