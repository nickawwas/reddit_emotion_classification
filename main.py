# pip3 install numpy matplotlib scikit-learn gensim nltk 

import json
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def plot_distribution(np_data, plt_axis, data_type):
    # Obtain number of occurences of each emotion or sentiment
    labels, frequency = np.unique(np_data, return_counts=True)

    # Plot emotions or sentiments in pie chart
    plt_axis.pie(frequency, labels=labels, autopct='%1.2f%%')
    plt_axis.set_title(f"{data_type} Distribution")

    return labels

# Open emotions dataset
emotions_dataset = 'fakeemotions.json'
with open(emotions_dataset, 'r') as file:
    # Read json data
    data = json.load(file)
    
    # Convert data to numpy array
    np_data = np.array(data)

    # Extract comments, emotions and sentiments
    comments = np_data[:, 0]
    emotions = np_data[:, 1]
    sentiments = np_data[:, 2]
    buckets = np_data[:, -2:]

    # Create 1 figure with 2 subplots
    _, (ax1, ax2) = plt.subplots(1, 2)

    emotions_labels = plot_distribution(emotions, ax1, "Emotions")
    sentiments_labels = plot_distribution(sentiments, ax2, "Sentiments")
    # plt.savefig(fname="post_distribution.pdf")

    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()
    frequency = vector.toarray().sum(axis=0)
    print(f"Vocabulary Size: {len(tokens)}")

    # Split dataset into training and testing split
    x_train, x_test, y_train, y_test = train_test_split(comments, buckets, train_size=0.8, test_size=0.2)
    # print("Training Comments: ", x_train)
    # print("Testing Comments: ", x_test)
    # print("Training Emotions: ", y_train)
    # print("Testing Emotions: ", y_test)

    #TODO: Find issue with training and testing data causing classifier to fail...
    '''
    # Train and test Multinomial Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test))
    print(y_test)

    # Train and test Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)


    # Train and test Multi-Layered Perceptron Classifier
    clf = MLPClassifier()
    clf.fit(x_train, y_train)


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