# pip3 install numpy pandas matplotlib scikit-learn gensim nltk 

import json
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural network import MLPClassifier


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

    # # Obtain number of occurences of each emotion
    # counts = pd.Series(emotions).value_counts()

    # # Plot emotions in pie chart
    # plt.pie(counts) # labels=counts.keys
    # plt.show()

    # # Obtain number of occurences of each emotion
    # sentiment_counts = pd.Series(sentiments).value_counts()

    # # Plot emotions in pie chart
    # plt.pie(sentiment_counts) # labels=counts.keys
    # plt.show()

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





# TODO: 
# def plot_data(data):
#     # Obtain number of occurences of each emotion
#     sentiment_counts = pd.Series(sentiments).value_counts()

#     # Plot emotions in pie chart
#     plt.pie(sentiment_counts) # labels=counts.keys
#     plt.show()