# pip3 install numpy matplotlib scikit-learn gensim nltk 

import json
import os
import pickle
import numpy as np

from Models import Models
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

EMOTIONS_DATASET = 'goemotions.json'
PERF_FILE = 'performance_explore.txt'

if __name__ == '__main__':
    models = Models(EMOTIONS_DATASET, 'models_explore', PERF_FILE, None)
    np_comments, np_emotions, np_sentiments = models.get_dataset()

    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()

    # Add vocabulary size to the performance sheet
    with open(PERF_FILE, 'w') as performance:
        performance.write(f'Vocabulary size: {len(tokens)}\n')

    # Perform the substeps of question 2.3 for 2 different splits: 95%/5% and 50%/50%
    test_cases = ['0.95', '0.5']

    for test_case in test_cases:
        if test_case == '0.95':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = models.get_train_test_split(comments_vector, np_emotions, np_sentiments, 0.95, 0.05, 0)
        elif test_case == '0.5':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = models.get_train_test_split(comments_vector, np_emotions, np_sentiments, 0.5, 0.5, 0)
        
        temp = Models(EMOTIONS_DATASET, 'models_explore', PERF_FILE, test_case)
        print('Multinominal Naive Bayes Classification For Emotions')
        temp.naive_bayes_classifier(comments_train, emotions_train)

        print('Multinominal Naive Bayes Classification For Sentiments')
        temp.naive_bayes_classifier(comments_train, sentiments_train)

        print('Decision Tree classification For Emotions')
        temp.decision_tree_classifier(comments_train, emotions_train)

        print('Decision Tree classification For Sentiments')
        temp.decision_tree_classifier(comments_train, sentiments_train)

        print('Perceptron classification For Emotions')
        temp.perceptron_classifier(comments_train, emotions_train)

        print('Perceptron classification For Sentiments')
        temp.perceptron_classifier(comments_train, sentiments_train)

        print('GridSearch Multinominal Naive Bayes Classification For Emotions')
        temp.top_mnb_classifier(comments_train, emotions_train)

        print('GridSearch Multinominal Naive Bayes Classification For Sentiments')
        temp.top_mnb_classifier(comments_train, sentiments_train)

        print('GridSearch Decision Tree classification For Emotions')
        temp.top_decision_tree_classifier(comments_train, emotions_train)

        print('GridSearch Decision Tree classification For Sentiments')
        temp.top_decision_tree_classifier(comments_train, sentiments_train)

        print('GridSearch Perceptron classification For Emotions')
        temp.top_perceptron_classifier(comments_train, emotions_train)

        print('GridSearch Perceptron classification For Sentiments')
        temp.top_perceptron_classifier(comments_train, sentiments_train)
