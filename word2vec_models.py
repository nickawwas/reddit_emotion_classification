import nltk
import numpy as np
import os
import json
import pickle
import gensim.downloader as loader

from Spinner import Spinner #TODO: Remove or change to custom 
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

nltk.download('punkt')

MODEL = 'models/word2vec/word2vec-google-news-300.model'
EMOTIONS_DATASET = 'goemotions.json'
# EMOTIONS_DATASET = 'fakeemotions.json'

def perceptron_classifier(comments, feature):
    clf = MLPClassifier(activation='logistic', learning_rate='adaptive', max_iter=50)
    clf.fit(comments, feature)
    return clf

def top_perceptron_classifier(comments, feature):
    params = {
        'solver': ['adam', 'sgd'],
        'activation': ['tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(50, 50, 50), (20, 40, 60)],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [50]
    }
    clf = GridSearchCV(MLPClassifier(), param_grid=params)
    clf.fit(comments, feature)
    return clf

def export_model(clf, classifier_type: str, feature_type: str):
    if not os.path.exists(f'models/word2vec'):
        os.mkdir(f'models/word2vec')
    
    pickle.dump(clf, open(f'models/word2vec/w2v_{classifier_type}_{feature_type}.model', 'wb'))

def report_results(clf, classifier_type: str, feature_type: str, comments, feature, is_gs: bool):
    prediction = clf.predict(comments)
    score = clf.score(comments, feature)

    print(f'Word2Vec {classifier_type} {feature_type} Score: {score}')

    # create performance file entry
    with open('performance_w2v.txt', 'a') as performance:
        performance.writelines(f'\n{classifier_type} - classifying {feature_type}: \n\nParams: {clf} \n\nScore {score}\n')
        performance.writelines('\nConfusion Matrix:\n')
        performance.writelines(f'{np.array2string(confusion_matrix(feature, prediction))}\n')        
        performance.writelines('\nClassification Report:\n')
        performance.writelines(f'{classification_report(feature, prediction, zero_division=1)}\n')
        
        if is_gs:
            performance.writelines(f'Best parameters: {clf.best_params_}\n')
            performance.writelines(f'Best score: {clf.best_score_}\n\n')

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
        for key in tokens:
            if not model.has_index_for(key):
                missed_keys += 1
        arr = model.get_mean_vector(tokens, ignore_missing=True) #later, try to use ignore missing false to catch KeyError to count hit rate
        embeddings.append(arr.tolist())

    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(embeddings, np_emotions, np_sentiments, train_size=0.8, test_size=0.2)

    with open('performance_w2v.txt', 'a') as performance:
        performance.write(f'Number of tokens in the dataset (excluding punctuation): {num_tokens}')
        hit_rate = (num_tokens - missed_keys) / num_tokens
        performance.write(f'\nOverall hit rate: {hit_rate}\n')

    print('Word2Vec Perceptron Classification For Emotions')
    with Spinner():
        w2v_mlp_emotions = perceptron_classifier(comments_train, emotions_train)
    report_results(w2v_mlp_emotions, 'Perceptron', 'Emotions', comments_test, emotions_test, False)    
    export_model(w2v_mlp_emotions, 'Perceptron', 'emotions')

    print('Word2Vec Perceptron Classification For Sentiments')
    with Spinner():
        w2v_mlp_sentiments = perceptron_classifier(comments_train, sentiments_train)
    report_results(w2v_mlp_sentiments, 'Perceptron', 'Sentiments', comments_test, sentiments_test, False)    
    export_model(w2v_mlp_sentiments, 'Perceptron', 'sentiments')

    print('Word2Vec GridSearch Perceptron classification For Emotions')
    # with Spinner():
    w2v_top_mlp_emotions = top_perceptron_classifier(comments_train, emotions_train)
    report_results(w2v_top_mlp_emotions, 'GridSearch_Perceptron', 'Emotions', comments_test, emotions_test, True)
    export_model(w2v_top_mlp_emotions, 'GridSearch_Perceptron', 'emotions')

    print('Word2Vec GridSearch Perceptron classification For Sentiments')
    # with Spinner():
    w2v_top_mlp_sentiments = top_perceptron_classifier(comments_train, sentiments_train)
    report_results(w2v_top_mlp_sentiments, 'GridSearch_Perceptron', 'Sentiments', comments_test, sentiments_test, True)
    export_model(w2v_top_mlp_sentiments, 'GridSearch_Perceptron', 'sentiments')
