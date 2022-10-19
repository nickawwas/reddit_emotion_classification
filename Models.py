import json
import os
import pickle
import numpy as np
import gensim.downloader as loader

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.models import KeyedVectors
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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
        plt.savefig(fname=f'charts/post_distribution_{style}chart.pdf')

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

    def naive_bayes_classifier(self, comments, feature, type: str):
        # Train and test Multinomial Naive Bayes Classifier
        clf = MultinomialNB()
        clf.fit(comments, feature)

        self.report_results(clf, 'Naive_Bayes_Classifier', type, comments, feature, False)
        self.export_model(clf, 'Naive_Bayes_Classifier', type.lower())
        return clf

    def decision_tree_classifier(self, comments, feature, type: str):
        clf = DecisionTreeClassifier()
        clf.fit(comments, feature)
        self.report_results(clf, 'Decision_Tree_Classifier', type, comments, feature, False)
        self.export_model(clf, 'Decision_Tree_Classifier', type.lower())
        return clf

    def perceptron_classifier(self, comments, feature, type: str):
        clf = MLPClassifier(max_iter=50)
        clf.fit(comments, feature)

        self.report_results(clf, 'Perceptron', type, comments, feature, False)    
        self.export_model(clf, 'Perceptron', type.lower())
        return clf

    def top_mnb_classifier(self, comments, feature, type: str):
        # n_jobs param value -1: allows utilization of maximum processors
        clf = GridSearchCV(MultinomialNB(), param_grid={'alpha': [0, 0.1, 0.5, 1.0]}, n_jobs=-1)
        clf.fit(comments, feature)
        
        self.report_results(clf, 'GridSearch_MNB', type, comments, feature, True)
        self.export_model(clf, 'GridSearch_MNB', type.lower())
        return clf

    def top_decision_tree_classifier(self, comments, feature, params, type: str):
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params)
        clf.fit(comments, feature)

        self.report_results(clf, 'GridSearch_DCT', type, comments, feature, True)
        self.export_model(clf, 'GridSearch_DCT', type.lower())
        return clf

    def top_perceptron_classifier(self, comments, feature, params, type: str):
        clf = GridSearchCV(MLPClassifier(), param_grid=params)
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
            cfm = np.array2string(confusion_matrix(feature, prediction))
            performance.writelines(f'{cfm}\n')
            self.output_confusion_matrix(feature, prediction, classifier_type, feature_type)
            performance.writelines('\nClassification Report:\n')
            performance.writelines(f'{classification_report(feature, prediction, zero_division=1)}\n')
            
            if is_gs:
                performance.writelines(f'Best parameters: {clf.best_params_}\n')
                performance.writelines(f'Best score: {clf.best_score_}\n\n')

    def output_confusion_matrix(self, feature, prediction, classifier_type, feature_type):
        if not os.path.exists(f'{self.export_path}/{classifier_type}'):
            os.makedirs(f'{self.export_path}/{classifier_type}')
        ConfusionMatrixDisplay.from_predictions(y_true=feature, y_pred=prediction, cmap='viridis')
        plt.gcf().set_size_inches(20, 13)
        plt.savefig(f'{self.export_path}/{classifier_type}/{classifier_type}_confusion_matrix_{feature_type}_{self.test_case}.pdf')

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
        return pickle.load(open(model_path, 'rb'))

    def get_train_test_split(self, vector, emotions, sentiments, train):
        return train_test_split(vector, emotions, sentiments, train_size=train)
