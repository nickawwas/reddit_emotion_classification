from pyexpat import model
import nltk
import numpy as np
import os
import json
import pickle
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as loader
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')

MODEL = 'models/word2vec/word2vec-google-news-300.model'
EMOTIONS_DATASET = 'goemotions.json'

if __name__ == '__main__':
    if not os.path.exists(MODEL):
        word2vec = loader.load('word2vec-google-news-300')
        pickle.dump(word2vec, open(MODEL, 'wb'))
    
    # tokenize string
    data = None
    with open(EMOTIONS_DATASET, 'r') as file:
        # Read json data
        data = json.load(file)

    np_data = np.array(data)
    np_comments = np_data[:, 0]
    tokens = [word_tokenize(t) for t in sent_tokenize(np_comments)]

    print(tokens)
    exit()
    # tokenized = word_tokenize(data)
    # print(tokenized)
    # print(len(tokenized))
