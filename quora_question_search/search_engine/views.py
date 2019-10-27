import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from warnings import filterwarnings
filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(3)
tf.disable_v2_behavior()

import sys
import sqlite3
import pandas as pd
import numpy as np
from os import path
from tqdm import tqdm
from django.db import connection
from django.shortcuts import render
from scipy.sparse import csr_matrix
from warnings import filterwarnings
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from .elmo_helpers import load_elmo_embeddings, get_elmo_vectors
from keras.models import Model
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer, load_vocabulary, get_checkpoint_paths
from gensim.models.keyedvectors import KeyedVectors, Word2VecKeyedVectors

main_logger = logging.getLogger()
main_logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
main_logger.addHandler(stream_handler)
file_handler = logging.FileHandler("logs.txt")
file_handler.setLevel(logging.DEBUG)
main_logger.addHandler(file_handler)

main_logger.info("Make sure that you have downloaded pre-trained models!")

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


def build_db():
    main_logger.info("Reading data...")
    df = pd.read_csv(path.join("..", "quora_question_pairs_rus.csv"))
    corpus = list(set(df["question1"])) + list(set(df["question2"]))
    main_logger.info("Data successfully loaded!")
    conn = sqlite3.connect("quora_question_pairs_rus.db")
    db = conn.cursor()
    db.execute("""
              CREATE TABLE text_corpora
              (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              text TEXT NOT NULL,
              text_lemmatized TEXT NOT NULL);
              """)
    main_logger.info("Creating the database...")
    for text in tqdm(corpus):
        db.execute("""
                   INSERT INTO text_corpora
                   (text, text_lemmatized)
                   VALUES (?, ?);
                   """, (text, " ".join(lemmatize(text))))
        conn.commit()
    conn.close()
    main_logger.info("Database creation finished.")
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
        super(TfIdfSearch, self).__init__()
        self.data = self.load_data()
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        if not path.isfile("tfidf_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("tfidf_matrix.npy")

    def fit_transform(self):
        main_logger.info("Vectorizing texts...")
        TF = self.vectorizer.fit_transform(self.texts)
        main_logger.info("Computing IDF...")
        IDF = np.array([((TF.getnnz(axis=1)).sum() - y) / y
                        for y in TF.nonzero()[0]])
        main_logger.info("Building TF-IDF matrix...")
        TF_IDF = np.matmul(TF.transpose(), IDF)
        self.matrix = csr_matrix((TF_IDF, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        np.save("tfidf_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        main_logger.info("Searching...")
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
        super(BM25Search, self).__init__()
        self.data = self.load_data()
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        if not path.isfile("bm25_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("bm25_matrix.npy")

    def fit_transform(self):
        main_logger.info("Vectorizing texts...")
        TF = self.vectorizer.fit_transform(self.texts)
        main_logger.info("Computing IDF...")
        IDF = np.array([((TF.getnnz(axis=1)).sum() - y) / y
                        for y in TF.nonzero()[0]])
        main_logger.info("Building BM25 matrix...")
        BM25 = IDF * TF.data * 3 / (TF.data + 3)
        self.matrix = csr_matrix((BM25, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        np.save("bm25_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        main_logger.info("Searching...")
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
        super(Word2VecSearch, self).__init__()
        self.data = self.load_data()
        self.model = self.load_model()
        main_logger.info("Model successfully loaded!")
        if not path.isfile("word2vec_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("word2vec_matrix.npy")

    def load_model(self):
        main_logger.info("Loading Word2Vec model...")
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
        main_logger.info("Building Word2Vec matrix...")
        self.matrix = np.zeros((100000, self.model.vector_size))
        for i in tqdm(range(100000)):
            self.matrix[i] = self.transform(self.data[i])
        np.save("word2vec_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def search(self, query):
        main_logger.info("Searching...")
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


class FastTextSearch(SearchEngine):
    def __init__(self):
        super(FastTextSearch, self).__init__()
        self.data = self.load_data()
        self.model = self.load_model()
        main_logger.info("Model successfully loaded!")
        if not path.isfile("fasttext_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("fasttext_matrix.npy")

    def load_model(self):
        main_logger.info("Loading FastText model...")
        return KeyedVectors.load(path.join("..",
                                           "model_fasttext",
                                           "model.model"))

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
        main_logger.info("Building FastText matrix...")
        self.matrix = np.zeros((100000, self.model.vector_size))
        for i in tqdm(range(100000)):
            self.matrix[i] = self.transform(self.data[i])
        np.save("fasttext_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def search(self, query):
        main_logger.info("Searching...")
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


class ELMOSearch(SearchEngine):
    def __init__(self):
        super(ELMOSearch, self).__init__()
        self.data = self.load_data()
        self.batcher, self.ids, self.input = self.load_model()
        main_logger.info("Model successfully loaded!")
        if not path.isfile("elmo_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("elmo_matrix.npy")

    def load_model(self):
        main_logger.info("Loading ELMO model...")
        tf.reset_default_graph()
        return load_elmo_embeddings(path.join("..", "model_elmo"))

    def transform(self, text):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            text_vec = np.transpose(np.mean(get_elmo_vectors(sess, [text],
                                                             self.batcher,
                                                             self.ids,
                                                             self.input
                                                             ))).flatten()
        return text_vec

    def fit_transform(self):
        main_logger.info("Building ELMO matrix...")
        self.matrix = np.zeros((0, 1024))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer)
            for i in tqdm(range(0, 100000, 75)):
                self.matrix = np.vstack((self.matrix,
                                         np.mean(get_elmo_vectors(
                                             sess, self.data[i:i + 75],
                                             self.batcher, self.ids,
                                             self.input), axis=1)))
        np.save("elmo_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def search(self, query):
        main_logger.info("Searching...")
        result = np.matmul(self.matrix, self.transform(lemmatize(query)))
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


class RuBERTSearch(SearchEngine):
    def __init__(self):
        super(RuBERTSearch, self).__init__()
        self.data = self.load_data()
        self.model, self.vocab, self.tokenizer = self.load_model()
        main_logger.info("Model successfully loaded!")
        if not path.isfile("bert_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = self.load_matrix("bert_matrix.npy")

    def load_model(self):
        main_logger.info("Loading RuBERT model...")
        tf.reset_default_graph()
        paths = get_checkpoint_paths(path.join("..", "model_bert"))
        inputs = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint, seq_len=50)
        outputs = MaskedGlobalMaxPool1D(name="Pooling")(inputs.output)
        vocab = load_vocabulary(paths.vocab)
        return Model(inputs=inputs.inputs,
                     outputs=outputs), vocab, Tokenizer(vocab)

    def fit_transform(self):
        main_logger.info("Building RuBERT matrix...")
        self.matrix = np.zeros((100000, 768))
        segments = np.array([[0 for i in range(50)]])
        for index, text in enumerate(self.data):
            tokens = self.tokenizer.tokenize(" ".join(text))[:50]
            idxs = np.array([[self.vocab[token] for token in tokens]
                             + [0 for i in range(50 - len(tokens))]])
            self.matrix[index] = self.model.predict([idxs, segments])[0]
        np.save("bert_matrix.npy", self.matrix)
        main_logger.info("Matrix creation finished.")

    def search(self, query):
        main_logger.info("Searching...")
        segments = np.array([[0 for i in range(50)]])
        tokens = self.tokenizer.tokenize(" ".join(lemmatize(query)))[:50]
        idxs = np.array([[self.vocab[token] for token in tokens]
                         + [0 for i in range(50 - len(tokens))]])
        query_vec = self.model.predict([idxs, segments])[0]
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
        if not results:
            failed_search = True
    return render(request, "main.html", {"results": results,
                  "failed_search": failed_search})
