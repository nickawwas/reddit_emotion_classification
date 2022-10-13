# pip3 install numpy matplotlib scikit-learn gensim nltk 

from dataclasses import dataclass
import json
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz 
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# EMOTIONS_DATASET = 'fakeemotions.json'
EMOTIONS_DATASET = 'goemotions.json'

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

def naive_bayers_classifier(comments, feature):
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
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(comments, feature)
    return clf

def top_mnb_classifier():
    return

def top_decision_tree_classifier():
    return

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
    mnb_emotions = naive_bayers_classifier(comments_train, emotions_train)
    print(f'MNB Emotions Prediction: {mnb_emotions.predict(comments_test)}')
    print(f'MNB Emotions Test: {emotions_test}')
    print(f'Score: {mnb_emotions.score(comments_test, emotions_test)}')

    # Multinominal Naive Bayes classification for sentiments:
    mnb_sentiments = naive_bayers_classifier(comments_train, sentiments_train)
    print(f'MNB Sentiments Prediction: {mnb_sentiments.predict(comments_test)}')
    print(f'MNB Sentiment Test: {sentiments_test}')
    print(f'Score: {mnb_sentiments.score(comments_test, sentiments_test)}')

    # Decision Tree classification for emotions:
    dct_emotions = decision_tree_classifier(comments_train, emotions_train)
    print(f'DCT Emotions Prediction: {dct_emotions.predict(comments_test)}')
    print(f'DCT Emotions Test: {emotions_test}')
    print(f'Score: {dct_emotions.score(comments_test, emotions_test)}')

    # Decision Tree classification for sentiments:
    dct_sentiments = decision_tree_classifier(comments_train, sentiments_train)
    print(f'DCT Sentiments Prediction: {dct_sentiments.predict(comments_test)}')
    print(f'DCT Sentiment Test: {sentiments_test}')
    print(f'Score: {dct_sentiments.score(comments_test, sentiments_test)}')
 
    mlp_emotions = perceptron_classifier(comments_train, emotions_train)
    print(mlp_emotions.predict_proba(comments_test))
    print(f'MLP Emototions Prediction: {mlp_emotions.predict(comments_test)}')
    print(f'MLP Emotions Test: {emotions_test}')
    print(f'Score: {mlp_emotions.score(comments_test, emotions_test)}')

    mlp_sentiments = perceptron_classifier(comments_train, sentiments_train)
    print(mlp_sentiments.predict_proba(comments_test))
    print(f'MLP Sentiment Prediction: {mlp_sentiments.predict(comments_test)}')
    print(f'MLP Sentiment Test: {sentiments_test}')
    print(f'Score: {mlp_sentiments.score(comments_test, sentiments_test)}')

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

