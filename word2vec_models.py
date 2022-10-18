import nltk
import os
import numpy as np
import pickle
import gensim.downloader as loader

from Models import Models
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')

MODEL = 'models/word2vec/word2vec-google-news-300.model'
EMOTIONS_DATASET = 'goemotions.json'
PERF_FILE = 'performance_w2v.txt'

if __name__ == '__main__':  
    models = Models(EMOTIONS_DATASET, 'models/word2vec', PERF_FILE, '0.8')
    model = models.import_model('models/word2vec', 'word2vec-google-news-300')

    np_comments, np_emotions, np_sentiments = models.get_dataset()

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
        for key in tokens:
            if not model.has_index_for(key):
                missed_keys += 1
        arr = model.get_mean_vector(tokens, ignore_missing=True)
        embeddings.append(arr.tolist())

    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = models.get_train_test_split(embeddings, np_emotions, np_sentiments, 0.8)

    with open(PERF_FILE, 'w') as performance:
        performance.write(f'Number of tokens in the dataset (excluding punctuation): {num_tokens}')
        hit_rate = (num_tokens - missed_keys) / num_tokens
        performance.write(f'\nOverall hit rate: {hit_rate}\n')

    print('Word2Vec Perceptron Classification For Emotions')
    models.perceptron_classifier(comments_train, emotions_train, 'Emotions')

    print('Word2Vec Perceptron Classification For Sentiments')
    models.perceptron_classifier(comments_train, sentiments_train, 'Sentiments')
    
    params = {
        'solver': ['adam', 'sgd'],
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(50, 50, 50), (20, 40, 60)],
        'max_iter': [50, 75]
    }
    print('Word2Vec GridSearch Perceptron classification For Emotions')
    models.top_perceptron_classifier(comments_train, emotions_train, params, 'Emotions')

    print('Word2Vec GridSearch Perceptron classification For Sentiments')
    models.top_perceptron_classifier(comments_train, sentiments_train, params, 'Sentiments')
