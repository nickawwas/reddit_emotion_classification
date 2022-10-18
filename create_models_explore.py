from Models import Models
from sklearn.feature_extraction.text import CountVectorizer

EMOTIONS_DATASET = 'goemotions.json'
PERF_FILE = 'performance_explore.txt'

if __name__ == '__main__':
    np_comments, np_emotions, np_sentiments = Models.get_dataset()

    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()

    # Add vocabulary size to the performance sheet
    with open(PERF_FILE, 'w') as performance:
        performance.write(f'Vocabulary size: {len(tokens)}\n')

    # Perform the substeps of question 2.3 for 2 different splits: 95%/5% and 50%/50%
    test_cases = ['0.95', '0.5', '0.25']

    for test_case in test_cases:
        models = Models(EMOTIONS_DATASET, 'models_explore', PERF_FILE, test_case)

        test_case = "{:.2f}".format(float(test_case))
        comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = models.get_train_test_split(comments_vector, np_emotions, np_sentiments, test_case, 1 - test_case, 0)
        
        print('Multinominal Naive Bayes Classification For Emotions')
        models.naive_bayes_classifier(comments_train, emotions_train, 'Emotions')
        
        print('Multinominal Naive Bayes Classification For Sentiments')
        models.naive_bayes_classifier(comments_train, sentiments_train, 'Sentiments')

        print('Decision Tree classification For Emotions')
        models.decision_tree_classifier(comments_train, emotions_train, 'Emotions')

        print('Decision Tree classification For Sentiments')
        models.decision_tree_classifier(comments_train, sentiments_train, 'Sentiments')

        print('Perceptron classification For Emotions')
        models.perceptron_classifier(comments_train, emotions_train, 'Emotions')

        print('Perceptron classification For Sentiments')
        models.perceptron_classifier(comments_train, sentiments_train, 'Sentiments')

        print('GridSearch Multinominal Naive Bayes Classification For Emotions')
        models.top_mnb_classifier(comments_train, emotions_train, 'Emotions')

        print('GridSearch Multinominal Naive Bayes Classification For Sentiments')
        models.top_mnb_classifier(comments_train, sentiments_train, 'Sentiments')

        print('GridSearch Decision Tree classification For Emotions')
        models.top_decision_tree_classifier(comments_train, emotions_train, 'Emotions')

        print('GridSearch Decision Tree classification For Sentiments')
        models.top_decision_tree_classifier(comments_train, sentiments_train, 'Sentiments')

        print('GridSearch Perceptron classification For Emotions')
        models.top_perceptron_classifier(comments_train, emotions_train, 'Emotions')

        print('GridSearch Perceptron classification For Sentiments')
        models.top_perceptron_classifier(comments_train, sentiments_train, 'Sentiments')
