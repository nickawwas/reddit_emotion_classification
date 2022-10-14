from pyexpat import model
import nltk
import numpy as np
import os
import json
import pickle
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as loader
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
nltk.download('punkt')

MODEL = 'models/word2vec/word2vec-google-news-300.model'
EMOTIONS_DATASET = 'goemotions.json'

if __name__ == '__main__':
    if not os.path.exists(MODEL):
        word2vec = loader.load('word2vec-google-news-300')
        os.mkdir('models/word2vec')
        pickle.dump(word2vec, open(MODEL, 'wb'))
    
    # tokenize string
    data = None
    with open(EMOTIONS_DATASET, 'r') as file:
        # Read json data
        data = json.load(file)

    np_data = np.array(data)
    np_comments = np_data[:, 0]

    # Ignore any punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = []

    for comment in np_comments:
      tokens += [word_tokenize(token) for token in tokenizer.tokenize(comment)]
    
    # do we need to remove tokens like commas, periods, apostrophes etc?
    num_tokens = 0
    for sentence in tokens:
        num_tokens += len(sentence)

    print(f'Number of tokens in the training set (excluding punctuation): {num_tokens}')
    exit()

    # tokenized = word_tokenize(data)
    # print(tokenized)
    # print(len(tokenized))
