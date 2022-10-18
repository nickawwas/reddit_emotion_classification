import json
import os
import pickle
import numpy as np
import graphviz 
import gensim.downloader as loader

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.models import KeyedVectors
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

class Models:
    dataset  = 'goemotions.json'
    export_path = 'models'
    perf_file = 'performance.txt'
    test_case = '0.8'

    def __init__(self, dataset, export_path, perf_file, test_case):
        self.dataset = 'goemotions.json' if dataset == None else dataset
        self.export_path = 'models' if export_path == None else export_path
        self.perf_file = 'performance.txt' if perf_file == None else perf_file
        self.test_case = '0.8' if test_case == None else test_case

    def get_dataset(self):
        with open(self.dataset, 'r') as file:
            # Read json data
            data = json.load(file)
        np_data = np.array(data)

        return np_data[:, 0], np_data[:, 1], np_data[:, 2]

    def plot_data(self, emotions, sentiments, style: str):
        # Create 1 figure with 2 subplots
        _, (ax1, ax2) = plt.subplots(1, 2, layout="constrained")
        
        self.plot_distribution(emotions, ax1, "Emotions", style)
        self.plot_distribution(sentiments, ax2, "Sentiments", style)
        plt.savefig(fname="charts/post_distribution_barchart.pdf")

    def plot_distribution(self, np_data, plt_axis, data_type, style: str):
        # Obtain number of occurences of each emotion or sentiment
        labels, frequency = np.unique(np_data, return_counts=True)

        if style == 'bar':
            # Plot emotions or sentiments in bar chart
            plt_axis.bar(labels, frequency)
            plt_axis.tick_params(labelrotation = 90)
            plt_axis.set_xlabel(data_type)
            plt_axis.set_ylabel("Frequency")
            plt_axis.set_title(f"{data_type} Distribution")
        else:
            # Plot emotions or sentiments in pie chart
            plt_axis.pie(frequency, labels=labels, rotatelabels = True, autopct='%1.2f%%')
            plt_axis.set_title(f"{data_type} Distribution")
        return labels

    def render_graph(self, dtc):
        # dot_data = tree.export_graphviz(dtc, out_file=None,
        # # feature_names=['Outlook', 'Temperature', 'Humidity', 'Windy'],
        # # class_names=['Don\'t Play', 'Play'],
        # filled=True, rounded=True) 
        # graph = graphviz.Source(dot_data) 
        # graph.render("mytree")
        return

    def naive_bayes_classifier(self, comments, feature, type: str):
        # Train and test Multinomial Naive Bayes Classifier
        clf = MultinomialNB()
        clf.fit(comments, feature)

        self.report_results(clf, 'Naive Bayes Classifier', type, comments, feature, False)
        self.export_model(clf, 'mnb', type.lower())
        return clf

    def decision_tree_classifier(self, comments, feature, type: str):
        clf = DecisionTreeClassifier()
        clf.fit(comments, feature)

        self.render_graph(clf) #TODO

        self.report_results(clf, 'Decision Tree Classifier', type, comments, feature, False)
        self.export_model(clf, 'dct', type.lower())
        return clf

    def perceptron_classifier(self, comments, feature, type: str):
        clf = MLPClassifier(max_iter=50)
        clf.fit(comments, feature)

        self.report_results(clf, 'Perceptron', type, comments, feature, False)    
        self.export_model(clf, 'mlp', type.lower())
        return clf

    def top_mnb_classifier(self, comments, feature, type: str):
        # n_jobs param value -1: allows utilization of maximum processors
        clf = GridSearchCV(MultinomialNB(), param_grid={'alpha': [0.01, 0.1, 0.5, 1.0]}, n_jobs=-1)
        clf.fit(comments, feature)
        
        self.report_results(clf, 'GridSearch_MNB', type, comments, feature, True)
        self.export_model(clf, 'GridSearch_MNB', type.lower())
        return clf

    def top_decision_tree_classifier(self, comments, feature, params, type: str):
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 8],
            'min_samples_split': [2, 3, 4]
        }
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params)
        clf.fit(comments, feature)

        self.report_results(clf, 'GridSearch_DCT', type, comments, feature, True)
        self.export_model(clf, 'GridSearch_DCT', type.lower())
        return clf

    def top_perceptron_classifier(self, comments, feature, params, type: str):
        params = {
            'solver': ['adam', 'sgd'],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],
            'hidden_layer_sizes': [(30, 30, 30), (10, 30, 50)],
            'max_iter': [50]
        }
        clf = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1)
        clf.fit(comments, feature)

        self.report_results(clf, 'GridSearch_MLP', type, comments, feature, True)
        self.export_model(clf, 'GridSearch_MLP', type.lower())
        return clf

    def report_results(self, clf, classifier_type: str, feature_type: str, comments, feature, is_gs: bool):
        prediction = clf.predict(comments)
        score = clf.score(comments, feature)

        print(f'{classifier_type} {feature_type} Score: {score}')

        # create performance file entry
        with open(self.perf_file, 'a') as performance:
            performance.writelines(f'\n{classifier_type} - classifying {feature_type}: \n\nParams: {clf} \n\nScore {score}\n')
            performance.writelines('\nConfusion Matrix:\n')
            performance.writelines(f'{np.array2string(confusion_matrix(feature, prediction))}\n')        
            performance.writelines('\nClassification Report:\n')
            performance.writelines(f'{classification_report(feature, prediction, zero_division=1)}\n')
            
            if is_gs:
                performance.writelines(f'Best parameters: {clf.best_params_}\n')
                performance.writelines(f'Best score: {clf.best_score_}\n\n')

    def export_model(self, clf, classifier_type: str, feature_type: str):
        if not os.path.exists(f'{self.export_path}/{classifier_type}'):
            os.makedirs(f'{self.export_path}/{classifier_type}')
        
        pickle.dump(clf, open(f'{self.export_path}/{classifier_type}/{classifier_type}_{feature_type}_{self.test_case}.model', 'wb'))

    def import_model(self, model_path: str, model_name: str):
        if not os.path.exists(f'{model_path}/{model_name}.model'):
            model = loader.load(model_name)
            os.makedirs(model_path)
            pickle.dump(model, open(f'{model_path}/{model_name}.model', 'wb'))
            
        return KeyedVectors.load(f'{model_path}/{model_name}.model')

    def load_model(self, model_path):
        return pickle.load(model_path)

    def get_train_test_split(self, vector, emotions, sentiments, train, test, random: int):
        return train_test_split(vector, emotions, sentiments, train_size=train, test_size=test, random_state=random)
