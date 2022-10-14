from mmap import mmap
import gensim
import nltk
import numpy as np
import os
import json
import pickle
from gensim.models import KeyedVectors
import gensim.downloader as loader
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

nltk.download('punkt')

MODEL = 'models/word2vec/word2vec-google-news-300.model'
# EMOTIONS_DATASET = 'goemotions.json'
EMOTIONS_DATASET = 'fakeemotions.json'

def perceptron_classifier(comments, feature):
    # TODO: Remove max iter, default = 200
    clf = MLPClassifier()
    clf.fit(comments, feature)
    return clf

if __name__ == '__main__':
    word2vec = None
    if not os.path.exists(MODEL):
        word2vec = loader.load('word2vec-google-news-300')
        os.mkdir('models/word2vec')
        pickle.dump(word2vec, open(MODEL, 'wb'))
    
    model = KeyedVectors.load(MODEL)

    # tokenize string
    data = None
    with open(EMOTIONS_DATASET, 'r') as file:
        # Read json data
        data = json.load(file)

    np_data = np.array(data)
    np_comments = np_data[:, 0]    
    np_emotions = np_data[:, 1]
    np_sentiments = np_data[:, 2]
    
    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(np_comments, np_emotions, np_sentiments, train_size=0.8, test_size=0.2)

    # Ignore any punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    embeddings = []
    num_tokens = 0
    missed_keys = 0
    for comment in comments_train:
        tokens = tokenizer.tokenize(comment) #3.2
        num_tokens += len(tokens)
        try:
            embeddings.append(model.get_mean_vector(tokens, ignore_missing=False)) #3.3 done,  #3.4 we let ignore_missing be false, this raises KeyError, we can calculate the number of keyerrors raised toget the hit rate
        except KeyError:
            missed_keys += 1
            continue

    print(f'Number of tokens in the training set (excluding punctuation): {num_tokens}')
    hit_rate = (num_tokens - missed_keys) / num_tokens
    print(f'Training hit rate: {hit_rate}')
    print(embeddings)
    # print(perceptron_classifier(embeddings, emotions_train))


    
    
    exit()

    # tokenized = word_tokenize(data)
    # print(tokenized)
    # print(len(tokenized))
