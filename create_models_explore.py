# pip3 install numpy matplotlib scikit-learn gensim nltk 

import json
import os
import pickle
import numpy as np

from Spinner import Spinner #TODO: Remove or change to custom 
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

EMOTIONS_DATASET = 'goemotions.json'
NAIVE_BAYES = 'naive_bayes'
DECISION_TREE = 'decision_tree'
PERCEPTRON = 'perceptron'

def naive_bayes_classifier(comments, feature):
    # Train and test Multinomial Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(comments, feature)
    return clf

def decision_tree_classifier(comments, feature):
    clf = DecisionTreeClassifier()
    clf.fit(comments, feature)
    return clf

def perceptron_classifier(comments, feature):
    # TODO: Remove max iter, default = 200
    clf = MLPClassifier(max_iter=10)
    clf.fit(comments, feature)
    return clf

def top_mnb_classifier(comments, feature):
    # n_jobs param value -1: allows utilization of maximum processors
    clf = GridSearchCV(MultinomialNB(), param_grid={'alpha': [0.01, 0.1, 0.5, 1.0]}, n_jobs=-1)
    clf.fit(comments, feature)
    return clf

def top_decision_tree_classifier(comments, feature):
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 8],
        'min_samples_split': [2, 3, 4]
    }
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params, n_jobs=-1)
    clf.fit(comments, feature)
    return clf

def top_perceptron_classifier(comments, feature):
    # TODO: Remove max iter, default = 200
    params = {
        'solver': ['adam', 'sgd'],
        'activation': ['softmax', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30, 30, 30), (10, 30, 50)],
        'max_iter': [5]
    }
    clf = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1)
    clf.fit(comments, feature)
    return clf

def report_results(clf, classifier_type: str, feature_type: str, comments, feature, is_gs: bool, test_case: str):
    prediction = clf.predict(comments)
    score = clf.score(comments, feature)

    print(f'{classifier_type} {feature_type} Score: {score}')

    # create performance file entry
    with open('performance_explore.txt', 'a') as performance:
        performance.writelines(f'\n{classifier_type} - classifying {feature_type} (Test case {test_case}): \n\nParams: {clf} \n\nScore {score}\n')      
        performance.writelines('\nClassification Report:\n')
        performance.writelines(f'{classification_report(feature, prediction, zero_division=1)}\n')
        
        if is_gs:
            performance.writelines(f'Best parameters: {clf.best_params_}\n')
            performance.writelines(f'Best score: {clf.best_score_}\n\n')

def export_model(clf, classifier_type: str, feature_type: str, test_case: str):
    if not os.path.exists(f'models_explore/{test_case}/{classifier_type}'):
        os.mkdir(f'models_explore/{test_case}/{classifier_type}')
    
    pickle.dump(clf, open(f'models_explore/{test_case}/{classifier_type}/{classifier_type}_{feature_type}.model', 'wb'))

if __name__ == '__main__':
    data = None
    with open(EMOTIONS_DATASET, 'r') as file:
        # Read json data
        data = json.load(file)

    np_data = np.array(data)

    # Split data into separate lists
    np_comments = np_data[:, 0]
    np_emotions = np_data[:, 1]
    np_sentiments = np_data[:, 2]

    # Plot emotion and sentiment data
    plot_data(np_emotions, np_sentiments, 'bar')
    
    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()

    # Add vocabulary size to the performance sheet
    with open('performance.txt', 'a') as performance:
        performance.write(f'Vocabulary size: {len(tokens)}\n')

    # Perform the substeps of question 2.3 for 2 different splits of 80%/20%, 95%/5% and 50%/50%
    test_cases = ['0.8_1', '0.8_2', '0.95', '0.5']

    for test_case in test_cases:
    
        if test_case is '0.8_1':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(comments_vector, np_emotions, np_sentiments, train_size=0.8, test_size=0.2, random_state=1)
        elif test_case is '0.8_2':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(comments_vector, np_emotions, np_sentiments, train_size=0.8, test_size=0.2, random_state=2)
        if test_case is '0.95':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(comments_vector, np_emotions, np_sentiments, train_size=0.95, test_size=0.05)
        if test_case is '0.5':
            # Split dataset into training and testing split
            comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(comments_vector, np_emotions, np_sentiments, train_size=0.5, test_size=0.5)

        # IMPORTANT: Spinner class is not custom (using it to verify nothing is hanging, we can replace with a custom one later)
        # TODO: Replace or remove Spinner 
        
        print('Multinominal Naive Bayes Classification For Emotions')
        with Spinner():
            mnb_emotions = naive_bayes_classifier(comments_train, emotions_train)
        report_results(mnb_emotions, 'Naive Bayes Classifier', 'Emotions', comments_test, emotions_test, False, test_case)
        export_model(mnb_emotions, NAIVE_BAYES, 'emotions', test_case)

        print('Multinominal Naive Bayes Classification For Sentiments')
        with Spinner():
            mnb_sentiments = naive_bayes_classifier(comments_train, sentiments_train)
        report_results(mnb_sentiments, 'Naive Bayes Classifier', 'Sentiments', comments_test, sentiments_test, False, test_case)
        export_model(mnb_sentiments, NAIVE_BAYES, 'sentiments', test_case)

        print('Decision Tree classification For Emotions')
        with Spinner():
            dct_emotions = decision_tree_classifier(comments_train, emotions_train)
        report_results(dct_emotions, 'Decision Tree Classifier', 'Emotions', comments_test, emotions_test, False, test_case)
        export_model(dct_emotions, DECISION_TREE, 'emotions', test_case)

        print('Decision Tree classification For Sentiments')
        with Spinner():
            dct_sentiments = decision_tree_classifier(comments_train, sentiments_train)
        report_results(dct_sentiments, 'Decision Tree Classifier', 'Sentiments', comments_test, sentiments_test, False, test_case)
        export_model(dct_sentiments, DECISION_TREE, 'sentiments', test_case)

        print('Perceptron classification For Emotions')
        with Spinner():
            mlp_emotions = perceptron_classifier(comments_train, emotions_train)
        report_results(mlp_emotions, 'Perceptron', 'Emotions', comments_test, emotions_test, False, test_case)    
        export_model(mlp_emotions, PERCEPTRON, 'emotions', test_case)

        print('Perceptron classification For Sentiments')
        with Spinner():
            mlp_sentiments = perceptron_classifier(comments_train, sentiments_train)
        report_results(mlp_sentiments, 'Perceptron', 'Sentiments', comments_test, sentiments_test, False, test_case)
        export_model(mlp_sentiments, PERCEPTRON, 'sentiments', test_case)

        print('GridSearch Multinominal Naive Bayes Classification For Emotions')
        with Spinner():
            gs_mnb_emotions = top_mnb_classifier(comments_train, emotions_train)
        report_results(gs_mnb_emotions, 'GridSearch_MNB', 'Emotions', comments_test, emotions_test, True, test_case)
        export_model(gs_mnb_emotions, 'GridSearch_MNB', 'emotions', test_case)

        print('GridSearch Multinominal Naive Bayes Classification For Sentiments')
        with Spinner():
            gs_mnb_sentiments = top_mnb_classifier(comments_train, sentiments_train)
        report_results(gs_mnb_sentiments, 'GridSearch_MNB', 'Sentiments', comments_test, sentiments_test, True, test_case)
        export_model(gs_mnb_sentiments, 'GridSearch_MNB', 'sentiments', test_case)

        print('GridSearch Decision Tree classification For Emotions')
        with Spinner():
            gs_dct_emotions = top_decision_tree_classifier(comments_train, emotions_train)
        report_results(gs_dct_emotions, 'GridSearch_DCT', 'Emotions', comments_test, emotions_test, True, test_case)
        export_model(gs_dct_emotions, 'GridSearch_DCT', 'emotions', test_case)

        print('GridSearch Decision Tree classification For Sentiments')
        with Spinner():
            gs_dct_sentiments = top_decision_tree_classifier(comments_train, sentiments_train)
        report_results(gs_dct_sentiments, 'GridSearch_DCT', 'Sentiments', comments_test, sentiments_test, True, test_case)
        export_model(gs_dct_sentiments, 'GridSearch_DCT', 'sentiments', test_case)

        print('GridSearch Perceptron classification For Emotions')
        with Spinner():
            gs_prceptron_emotions = top_perceptron_classifier(comments_train, emotions_train)
        report_results(gs_prceptron_emotions, 'GridSearch_Perceptron', 'Emotions', comments_test, emotions_test, True, test_case)
        export_model(gs_prceptron_emotions, 'GridSearch_Perceptron', 'emotions', test_case)

        print('GridSearch Perceptron classification For Sentiments')
        with Spinner():
            gs_prceptron_sentiments = top_perceptron_classifier(comments_train, sentiments_train)
        report_results(gs_prceptron_sentiments, 'GridSearch_Perceptron', 'Sentiments', comments_test, sentiments_test, True, test_case)
        export_model(gs_prceptron_sentiments, 'GridSearch_Perceptron', 'sentiments', test_case)
