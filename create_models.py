from Models import Models
from sklearn.feature_extraction.text import CountVectorizer

EMOTIONS_DATASET = 'goemotions.json'
PERF_FILE = 'performance.txt'

if __name__ == '__main__':
    models = Models(EMOTIONS_DATASET, 'models', PERF_FILE, '0.8')
    np_comments, np_emotions, np_sentiments = models.get_dataset()

    # Plot emotion and sentiment data
    models.plot_data(np_emotions, np_sentiments, 'bar')
    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()

    # Add vocabulary size to the performance sheet
    with open(PERF_FILE, 'w') as performance:
        performance.write(f'Vocabulary size: {len(tokens)}\n')
    
    # Split dataset into training and testing split
    comments_train, comments_test, emotions_train, emotions_test, sentiments_train, sentiments_test = models.get_train_test_split(comments_vector, np_emotions, np_sentiments, 0.8, 0.2, 0)

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
