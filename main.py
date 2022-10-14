# pip3 install numpy matplotlib scikit-learn gensim nltk 

import json
import os
import pickle
from Spinner import Spinner
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz 
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# EMOTIONS_DATASET = 'fakeemotions.json'
EMOTIONS_DATASET = 'goemotions.json'
NAIVE_BAYES='naive_bayes'
DECISION_TREE='decision_tree'
PERCEPTRON='perceptron'

def bar_plot_distribution(np_data, plt_axis, data_type):
    # Obtain number of occurences of each emotion or sentiment
    labels, frequency = np.unique(np_data, return_counts=True)

    # Plot emotions or sentiments in bar chart
    plt_axis.bar(labels, frequency)
    plt_axis.tick_params(labelrotation = 90)
    plt_axis.set_xlabel(data_type)
    plt_axis.set_ylabel("Frequency")
    plt_axis.set_title(f"{data_type} Distribution")

    return labels

def plot_distribution(np_data, plt_axis, data_type):
    # Obtain number of occurences of each emotion or sentiment
    labels, frequency = np.unique(np_data, return_counts=True)

    # Plot emotions or sentiments in pie chart
    plt_axis.pie(frequency, labels=labels, rotatelabels = True, autopct='%1.2f%%')
    plt_axis.set_title(f"{data_type} Distribution")

    return labels

def plot_data(emotions, sentiments):
    # Create 1 figure with 2 subplots
    _, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
    
    bar_plot_distribution(emotions, ax1, "Emotions")
    bar_plot_distribution(sentiments, ax2, "Sentiments")

    plt.savefig(fname="post_distribution.pdf")

def render_graph(dtc):
    # dot_data = tree.export_graphviz(dtc, out_file=None,
    # # feature_names=['Outlook', 'Temperature', 'Humidity', 'Windy'],
    # # class_names=['Don\'t Play', 'Play'],
    # filled=True, rounded=True) 
    # graph = graphviz.Source(dot_data) 
    # graph.render("mytree")
    return

def naive_bayes_classifier(comments, feature):
    # Train and test Multinomial Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(comments, feature)
    return clf

def decision_tree_classifier(comments, feature):
    clf = DecisionTreeClassifier()
    clf.fit(comments, feature)

    render_graph(clf) #TODO
    return clf

def perceptron_classifier(comments, feature):
    clf = MLPClassifier(random_state=1, max_iter=100)
    clf.fit(comments, feature)
    return clf

def top_mnb_classifier():
    return

def top_decision_tree_classifier():
    return

def top_perceptron_classifier():
    return

def report_results(clf, classifier_type: str, feature_type: str, comments, feature):
    # print(f'{classfier_type} {feature_type} probability eastimate: {clf.predict_proba(comments_test)}')
    prediction = clf.predict(comments)
    score = clf.score(comments, feature)

    print(f'{classifier_type} {feature_type} Prediction: {prediction}')
    print(f'{classifier_type} {feature_type} Test: {feature}')
    print(f'Score: {score}')

    # create performance file entry
    with open('performance.txt', 'a') as performance:
        performance.writelines(f'\n{classifier_type}- classifying {feature_type}: Score {score}\n')
        performance.writelines('Confusion Matrix:\n')
        performance.writelines(np.array2string(confusion_matrix(feature, prediction)))        
        performance.writelines('\n\nClassification Report:\n')
        performance.writelines(f'{classification_report(feature, prediction, zero_division=1)}\n\n')

def export_model(clf, classifier_type: str, feature_type: str):
    if not os.path.exists(f'models/{classifier_type}'):
        os.mkdir(f'models/{classifier_type}')
    
    pickle.dump(clf, open(f'models/{classifier_type}/{classifier_type}_{feature_type}.sav', 'wb'))

if __name__ == '__main__':
    data = None
    with open(EMOTIONS_DATASET, 'r') as file:
        # Read json data
        data = json.load(file)
        np_data = np.array(data)

    np_comments = np_data[:, 0]
    np_emotions = np_data[:, 1]
    np_sentiments = np_data[:, 2]

    # Plot emotion and sentiment data
    plot_data(np_emotions, np_sentiments)

    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()    
    print(f"Vocabulary Size: {len(tokens)}")
    
    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = train_test_split(comments_vector, np_emotions, np_sentiments, train_size=0.8, test_size=0.2)

    # Multinominal Naive Bayes classification for emotions:

    # IMPORTANT: Spinner class is not custom (using it to verify nothing is hanging, we can replace with a custom one later)
    # TODO: Replace or remove Spinner 
    with Spinner():
        mnb_emotions = naive_bayes_classifier(comments_train, emotions_train)
    report_results(mnb_emotions, 'Naive Bayes Classifier', 'Emotions', comments_test, emotions_test)
    export_model(mnb_emotions, NAIVE_BAYES, 'emotions')

    with Spinner():
        # Multinominal Naive Bayes classification for sentiments:
        mnb_sentiments = naive_bayes_classifier(comments_train, sentiments_train)
    report_results(mnb_sentiments, 'Naive Bayes Classifier', 'Sentiments', comments_test, sentiments_test)
    export_model(mnb_sentiments, NAIVE_BAYES, 'sentiments')

    with Spinner():
        # Decision Tree classification for emotions:
        dct_emotions = decision_tree_classifier(comments_train, emotions_train)
    report_results(dct_emotions, 'Decision Tree Classifier', 'Emotions', comments_test, emotions_test)
    export_model(dct_emotions, DECISION_TREE, 'emotions')

    with Spinner():
        # Decision Tree classification for sentiments:
        dct_sentiments = decision_tree_classifier(comments_train, sentiments_train)
    report_results(dct_sentiments, 'Decision Tree Classifier', 'Sentiments', comments_test, sentiments_test)
    export_model(dct_sentiments, DECISION_TREE, 'sentiments')

    # with Spinner():
    #     # Perceptron classification for emotions:
    #     mlp_emotions = perceptron_classifier(comments_train, emotions_train)
    # report_results(mlp_emotions, 'MLP', 'Emotions', comments_test, emotions_test)

    # with Spinner():
    #     # Perceptron classification for sentiments:
    #     mlp_sentiments = perceptron_classifier(comments_train, sentiments_train)
    # report_results(mlp_sentiments, 'MLP', 'Sentiments', comments_test, sentiments_test)

    #TODO: GridSearchCV
    #TODO: Top-MLP

    '''


    # Train and test better perfomring MNB Classifier
    clf = GridSearchCV(
        MultinomialNB(), 
        #...
    )
    clf.fit(x_train, y_train)

    # Train and test better perfomring DT Classifier
    # TODO: 2 different values for max_depth, 3 different values for min_samples_split
    clf = GridSearchCV(
        DecisionTreeClassifier(), 
        #...
    )
    clf.fit(x_train, y_train)


    # Train and test better perfomring MLP Classifier
    # TODO: sigmoid, tanh, relu + identity for activation, 2 network architecture and Adam + stochastic gradient descent for solver
    clf = GridSearchCV(
        MLPClassifier(), 
        #...
    )
    clf.fit(x_train, y_train)
    '''

    # Save classification results for each in performance folder

