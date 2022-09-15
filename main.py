# pip3 install numpy matplotlib scikit-learn gensim nltk 

import json
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def plot_distribution(np_data, data_type):
    # Obtain number of occurences of each emotion or sentiment
    labels, frequency = np.unique(np_data, return_counts=True)

    # Plot emotions or sentiments in pie chart
    plt.pie(frequency, labels=labels, autopct='%1.2f%%')
    plt.title(f"{data_type} Distribution")
    plt.savefig(fname=f"{data_type}_distribution.pdf")

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

    '''
    # Create 1 figure with 2 subplots
    _, (ax1, ax2) = plt.subplots(1, 2)

    # Obtain number of occurences of each emotion
    emotions_labels, emotions_frequency = np.unique(emotions, return_counts=True)

    # Plot emotions in pie chart
    ax1.pie(emotions_frequency, labels=emotions_labels, autopct='%1.2f%%')
    ax1.set_title('Emotions Distribution')

    # Obtain number of occurences of each sentiment
    sentiments_labels, sentiments_frequency = np.unique(sentiments, return_counts=True)

    # Plot sentiments in pie chart
    ax2.pie(sentiments_frequency, labels=sentiments_labels, autopct='%1.2f%%')
    ax2.set_title('Sentiments Distribution')
    plt.savefig(fname="post_distribution.pdf")
    '''

    # Define vectorizer
    vectorizer = CountVectorizer(analyzer='word')
    vector = vectorizer.fit_transform(comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()
    frequency = vector.toarray().sum(axis=0)
    print(tokens, frequency)

    # Split dataset into training and testing split
    x_train, x_test, y_train, y_test = train_test_split(tokens, frequency, train_size=0.8, test_size=0.2, random_state=40)
    print(x_train, x_test, y_train, y_test)

    # Train and test Multinomial Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print(clf.predict(x_test[2:3]))
    print(y_test[2:3])