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
EMOTIONS_DATASET = 'goemotions.json'
# EMOTIONS_DATASET = 'fakeemotions.json'

def perceptron_classifier(comments, feature):
    clf = MLPClassifier(activation='logistic', learning_rate='adaptive')
    clf.fit(comments, feature)
    return clf

def top_perceptron_classifier(comments, feature):
    params = {
        'solver': ['adam', 'sgd'],
        'activation': ['softmax', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30, 30, 30), (10, 30, 50)],
        'max_iter': [5]
    }
    clf = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1)
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

    # Ignore any punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    embeddings = []
    num_tokens = 0
    missed_keys = 0
    for i, comment in enumerate(np_comments):
        tokens = tokenizer.tokenize(comment) #3.2
        num_tokens += len(tokens)
        if len(tokens) == 0: # handle cases where no chars are valid
            np_emotions = np.delete(np_emotions, i)
            np_sentiments = np.delete(np_sentiments, i)
            continue
        arr = model.get_mean_vector(tokens, ignore_missing=True) #later, try to use ignore missing false to catch KeyError to count hit rate
        embeddings.append(arr.tolist())

    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(embeddings, np_emotions, np_sentiments, train_size=0.8, test_size=0.2)

    print(f'Number of tokens in the training set (excluding punctuation): {num_tokens}')
    hit_rate = (num_tokens - missed_keys) / num_tokens
    print(f'Training hit rate: {hit_rate}')

    w2v_mlp = perceptron_classifier(comments_train, emotions_train)
    print(w2v_mlp)
    print(w2v_mlp.score(comments_test, emotions_test))
    prediction = w2v_mlp.predict(comments_test)
    print(prediction)
    print(emotions_test)
    
    # w2v_top_mlp = top_perceptron_classifier(comments_train, sentiments_train)
    

    exit()

