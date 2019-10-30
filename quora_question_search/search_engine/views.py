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
from tqdm import tqdm
from django.db import connection
from django.shortcuts import render
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer
from .elmo_helpers import load_elmo_embeddings, get_elmo_vectors
from smart_open import open
from keras.models import Model
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer, load_vocabulary, get_checkpoint_paths
from gensim.models.keyedvectors import KeyedVectors, Word2VecKeyedVectors

logging.info("Make sure that you have downloaded pre-trained models!")

m = MorphAnalyzer()


def lemmatize(text):
    return [m.parse(word)[0].normal_form
            for word in simple_word_tokenize(text)]


def tag(text):
    return [f"{m.parse(word)[0].normal_form}_{m.parse(word)[0].POS}"
            for word in simple_word_tokenize(text)]


def build_db():
    logging.info("Reading data...")
    with open("quora_question_pairs_rus.csv", "r") as file:
        df = pd.read_csv(file)
        df = df.dropna(subset=["question1", "question2"])
        corpus = list(set(df["question1"])) + list(set(df["question2"]))
    logging.info("Data successfully loaded!")
    conn = sqlite3.connect("quora_question_pairs_rus.db")
    db = conn.cursor()
    db.execute("""
              CREATE TABLE text_corpora
              (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              text TEXT NOT NULL,
              text_lemmatized TEXT NOT NULL,
              text_tagged TEXT NOT NULL);
              """)
    logging.info("Creating the database...")
    for text in tqdm(corpus):
        db.execute("""
                   INSERT INTO text_corpora
                   VALUES (?, ?, ?);
                   """, (text, " ".join(lemmatize(text)),
                         " ".join(tag(text))))
        conn.commit()
    conn.close()
    logging.info("Database creation finished.")


if not os.path.isfile("quora_question_pairs_rus.db"):
    build_db()


class SearchEngine():
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        logging.info("Loading data...")
        db = connection.cursor()
        db.execute("""
                   SELECT text_lemmatized FROM text_corpora LIMIT 100000
                   """)
        data = db.fetchall()
        db.close()
        logging.info("Data loaded!")
        return np.array([text[0] for text in data])

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
        if not os.path.isfile("tfidf_matrix.npz"):
            self.fit_transform()
        else:
            self.vectorizer.fit(self.data)
            self.matrix = load_npz("tfidf_matrix.npz")

    def fit_transform(self):
        logging.info("Vectorizing texts...")
        TF = self.vectorizer.fit_transform(self.data)
        logging.info("Computing IDFs...")
        IDF = np.array([np.log((100000 - y + 0.5) / (y + 0.5))
                        for y in TF.data])
        logging.info("Building TF-IDF matrix...")
        TF_IDF = TF.data * IDF
        self.matrix = csr_matrix((TF_IDF, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        save_npz("tfidf_matrix.npz", self.matrix)
        logging.info("Matrix creation finished.")

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        logging.info("Searching...")
        query_vec = self.transform([query])
        result = np.array((query_vec * self.matrix).todense())[0]
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        db = connection.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


class BM25Search(SearchEngine):  # b=0, k=2
    def __init__(self):
        super(BM25Search, self).__init__()
        self.data = self.load_data()
        self.vectorizer = CountVectorizer(tokenizer=lemmatize)
        if not os.path.isfile("bm25_matrix.npz"):
            self.fit_transform()
        else:
            self.vectorizer.fit(self.data)
            self.matrix = load_npz("bm25_matrix.npz")

    def fit_transform(self):
        logging.info("Vectorizing texts...")
        TF = self.vectorizer.fit_transform(self.data)
        N = (TF.getnnz(axis=1)).sum()
        logging.info("Computing IDFs...")
        IDF = np.array([np.log((N - y + 0.5) / (y + 0.5))
                        for y in TF.nonzero()[0]])
        logging.info("Building BM25 matrix...")
        BM25 = IDF * TF.data * 3 / (TF.data + 3)
        self.matrix = csr_matrix((BM25, TF.indices, TF.indptr),
                                 shape=TF.shape)
        self.matrix = self.matrix.transpose(copy=True)
        save_npz("bm25_matrix.npz", self.matrix)
        logging.info("Matrix creation finished.")

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def search(self, query):
        logging.info("Searching...")
        query_vec = self.transform([query])
        result = np.array((query_vec * self.matrix).todense())[0]
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        db = connection.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


class Word2VecSearch(SearchEngine):
    def __init__(self):
        super(Word2VecSearch, self).__init__()
        self.data = self.load_tagged_data()
        self.model = self.load_model()
        logging.info("Model successfully loaded!")
        if not os.path.isfile("word2vec_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = np.load("word2vec_matrix.npy")

    def load_tagged_data(self):
        logging.info("Loading data...")
        db = connection.cursor()
        db.execute("""
                   SELECT text_tagged FROM text_corpora LIMIT 100000
                   """)
        data = db.fetchall()
        db.close()
        logging.info("Data loaded!")
        return np.array([text[0] for text in data])

    def load_model(self):
        logging.info("Loading Word2Vec model...")
        return Word2VecKeyedVectors.load_word2vec_format(
            os.path.join("model_word2vec", "model.bin"), binary=True)

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
        logging.info("Building Word2Vec matrix...")
        self.matrix = np.zeros((100000, self.model.vector_size))
        for i in tqdm(range(100000)):
            self.matrix[i] = self.transform(self.data[i].split(" "))
        np.save("word2vec_matrix.npy", self.matrix)
        logging.info("Matrix creation finished.")

    def search(self, query):
        logging.info("Searching...")
        query_vec = np.transpose(self.transform(tag(query)))
        result = np.matmul(self.matrix, query_vec)
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        conn = sqlite3.connect("quora_question_pairs_rus.db")
        db = conn.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


class FastTextSearch(SearchEngine):
    def __init__(self):
        super(FastTextSearch, self).__init__()
        self.data = self.load_data()
        self.model = self.load_model()
        logging.info("Model successfully loaded!")
        if not os.path.isfile("fasttext_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = np.load("fasttext_matrix.npy")

    def load_model(self):
        logging.info("Loading FastText model...")
        return KeyedVectors.load(os.path.join("model_word2vec",
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
        logging.info("Building FastText matrix...")
        self.matrix = np.zeros((100000, self.model.vector_size))
        for i in tqdm(range(100000)):
            self.matrix[i] = self.transform(self.data[i].split(" "))
        np.save("fasttext_matrix.npy", self.matrix)
        logging.info("Matrix creation finished.")

    def search(self, query):
        logging.info("Searching...")
        query_vec = np.transpose(self.transform(lemmatize(query)))
        result = np.matmul(self.matrix, query_vec)
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        db = connection.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


class ELMOSearch(SearchEngine):
    def __init__(self):
        super(ELMOSearch, self).__init__()
        self.data = self.load_data()
        self.batcher, self.ids, self.input = self.load_model()
        logging.info("Model successfully loaded!")
        if not os.path.isfile("elmo_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = np.load("elmo_matrix.npy")

    def load_model(self):
        logging.info("Loading ELMO model...")
        tf.reset_default_graph()
        return load_elmo_embeddings("model_elmo")

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
        logging.info("Building ELMO matrix...")
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
        logging.info("Matrix creation finished.")

    def search(self, query):
        logging.info("Searching...")
        result = np.matmul(self.matrix, self.transform(lemmatize(query)))
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        db = connection.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


class RuBERTSearch(SearchEngine):
    def __init__(self):
        super(RuBERTSearch, self).__init__()
        self.data = self.load_data()
        self.model, self.vocab, self.tokenizer = self.load_model()
        logging.info("Model successfully loaded!")
        if not os.path.isfile("bert_matrix.npy"):
            self.fit_transform()
        else:
            self.matrix = np.load("bert_matrix.npy")

    def load_model(self):
        logging.info("Loading RuBERT model...")
        tf.reset_default_graph()
        paths = get_checkpoint_paths("model_bert")
        inputs = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint, seq_len=50)
        outputs = MaskedGlobalMaxPool1D(name="Pooling")(inputs.output)
        vocab = load_vocabulary(paths.vocab)
        return Model(inputs=inputs.inputs,
                     outputs=outputs), vocab, Tokenizer(vocab)

    def fit_transform(self):
        logging.info("Building RuBERT matrix...")
        self.matrix = np.zeros((100000, 768))
        segments = np.array([[0 for i in range(50)]])
        for index, text in tqdm(enumerate(self.data)):
            tokens = self.tokenizer.tokenize(" ".join(text))[:50]
            idxs = np.array([[self.vocab[token] for token in tokens]
                             + [0 for i in range(50 - len(tokens))]])
            self.matrix[index] = self.model.predict([idxs, segments])[0]
        np.save("bert_matrix.npy", self.matrix)
        logging.info("Matrix creation finished.")

    def search(self, query):
        logging.info("Searching...")
        segments = np.array([[0 for i in range(50)]])
        tokens = self.tokenizer.tokenize(" ".join(lemmatize(query)))[:50]
        idxs = np.array([[self.vocab[token] for token in tokens]
                         + [0 for i in range(50 - len(tokens))]])
        query_vec = self.model.predict([idxs, segments])[0]
        result = np.matmul(self.matrix, query_vec)
        indices = np.argsort(result)[::-1].tolist()[:10]
        best_match = []
        db = connection.cursor()
        for index in tqdm(indices):
            db.execute(f"""SELECT text from text_corpora
                                  WHERE id={index}""")
            text = db.fetchone()[0]
            best_match.append((text, result[index]))
        db.close()
        return best_match


def main(request):
    query = request.GET.get("query")
    model = request.GET.get("model")
    results = []
    failed_search = False
    if query and model:
        if model == "tf-idf":
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
        else:
            engine = None
        results = engine.search(query)
        if not results:
            failed_search = True
    return render(request, "main.html", {"results": results,
                  "failed_search": failed_search})
