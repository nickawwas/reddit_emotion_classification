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

    models.get_train_test_split(comments_vector, np_emotions, np_sentiments)

    params_dct = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 8],
        'min_samples_split': [2, 3, 4]
    }

    params_mlp = {
        'solver': ['adam', 'sgd'],
        'activation': ['logistic', 'tanh', 'relu', 'identity'],
        'hidden_layer_sizes': [(30, 30, 30), (10, 30, 50)],
        'max_iter': [10]
    }
    models.run_models(params_dct=params_dct, params_mlp=params_mlp)
