from Models import Models
from sklearn.feature_extraction.text import CountVectorizer

EMOTIONS_DATASET = 'goemotions.json'
PERF_FILE = 'performance_explore.txt'

if __name__ == '__main__':
    temp = Models(EMOTIONS_DATASET, 'models_explore', PERF_FILE, '0.8')
    np_comments, np_emotions, np_sentiments = temp.get_dataset()

    # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
    vectorizer = CountVectorizer()
    comments_vector = vectorizer.fit_transform(np_comments)

    # Obtain words and frequency
    tokens = vectorizer.get_feature_names_out()

    # Add vocabulary size to the performance sheet
    with open(PERF_FILE, 'w') as performance:
        performance.write(f'Vocabulary size: {len(tokens)}\n')


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

    # Perform the substeps of question 2.3 for 4 different splits: 80%/20%, 80%/20%, 95%/5% and 25%/75%
    test_cases = ['0.8', '0.8', '0.95', '0.25']
    rand = 0

    for test_case in test_cases:
        if test_case == '0.8':
            rand += 1
        else:
            rand = 0

        models = Models(EMOTIONS_DATASET, 'models_explore', PERF_FILE, test_case, rand)

        models.get_train_test_split(comments_vector, np_emotions, np_sentiments)

        models.run_models(params_dct=params_dct, params_mlp=params_mlp)

