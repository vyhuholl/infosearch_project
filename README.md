# infosearch_project
Final project for this course: https://github.com/hse-infosearch/infosearch. A search engine on the [**Quora question pairs** dataset](https://www.kaggle.com/loopdigga/quora-question-pairs-russian), using several different search models. <br>
To run the search, first download the [archive with pre-trained models](https://www.dropbox.com/s/wxijsqqwrx71q32/Pre-trained%20models.zip?dl=0) and unzip it to the *quora_question_search* directory. <br>
Source for the code to work with ELMO word embeddings (file *elmo_helpers.py*): https://github.com/ltgoslo/simple_elmo
Sources for pre-trained **Word2Vec**, **Fasttext** and **BERT** language models: <br>
<br>
* https://rusvectores.org/en/models/
* http://docs.deeppavlov.ai/en/master/features/models/bert.html
<br>
Search models used: <br>
<br>
* **TF-IDF**
* **BM25**
* **Word2Vec**
* **Fasttext**
* **ELMO**
* **BERT**
