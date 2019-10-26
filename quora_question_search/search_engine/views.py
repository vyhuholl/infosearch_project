import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
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
