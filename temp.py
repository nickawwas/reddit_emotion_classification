from Models import Models
import os

import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

dirs = ['models', 'models_explore']

def update_names(path):
    files = os.listdir(path)

    for file in files:
        splits = os.path.splitext(file)

        if splits[1] == '.model':
            path_struct = path.split('/')
            export_path = path_struct[0]
            perf_file = 'performance.txt'
            test_case = splits[0].split('_')
            test_case = test_case[len(test_case) - 2]
            test_case = test_case.replace('-', '.')

            classifier_type = path_struct[1]
            is_gs = False
            feature = None

            np_comments = emotions = sentiments = None
            rand = 0
            if len(path_struct) == 3:
                export_path += f'/{path_struct[1]}'
                perf_file = f'performance_{path_struct[1]}.txt'
                classifier_type = path_struct[2]
            elif export_path == 'models_explore':
                perf_file = 'performance_explore.txt'
                rand = splits[0].split('_')
                rand = rand[len(rand) - 1]

            feature_type = splits[0].split('_')
            feature_type = feature_type[len(feature_type) - 3]
            
            if splits[0].split('_')[0] == 'GridSearch':
                is_gs = True
            
            models = Models(export_path=export_path, perf_file=perf_file, test_case=test_case, rand=rand)
            model = models.load_model(f'{path}/{file}')

            np_comments, emotions, sentiments = models.get_dataset()

            if path_struct[1] == 'gigaword':
                keyed_vec = models.import_model(export_path, 'glove-wiki-gigaword-50')
                np_comments, emotions, sentiments = get_embeddings(keyed_vec, perf_file, np_comments, emotions, sentiments)
            elif path_struct[1] == 'twitter':
                keyed_vec = models.import_model(export_path, 'glove-twitter-25')
            elif path_struct[1] == 'word2vec':
                keyed_vec = models.import_model(export_path, 'word2vec-google-news-300')
            else:
                # Create count vectorizer and learn vocabulary from comments to obtain single feature vector
                vectorizer = CountVectorizer()
                np_comments = vectorizer.fit_transform(np_comments)

                # # Obtain words and frequency
                # tokens = vectorizer.get_feature_names_out()

            models.get_train_test_split(np_comments, emotions, sentiments)

            if feature_type == 'emotions':
                feature = models.emotions_test
            else:
                feature = models.sentiments_test

            models.report_results(model, classifier_type=classifier_type, feature_type=feature_type, feature=feature, is_gs=is_gs)
            models.export_model(model, classifier_type=classifier_type, feature_type=feature_type)

            print(path.split('/'))
            print(splits)
            # exit()
        
        os.remove(f'{path}/{file}')
        
def get_embeddings(keyed_vec, perf_file, np_comments, np_emotions, np_sentiments):
    # Ignore any punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    embeddings = []
    num_tokens = 0
    missed_keys = 0
    for i, comment in enumerate(np_comments):
        tokens = tokenizer.tokenize(comment) #3.2
        num_tokens += len(tokens)
        if len(tokens) == 0: # handle cases where no chars are valid
            np_emotions = np.delete(np_emotions, i)
            np_sentiments = np.delete(np_sentiments, i)
            continue
        for key in tokens:
            if not keyed_vec.has_index_for(key):
                missed_keys += 1
        arr = keyed_vec.get_mean_vector(tokens, ignore_missing=True)
        embeddings.append(arr.tolist())
        
    return embeddings, np_emotions, np_sentiments

if __name__ == '__main__':
    # reprint graphs and model names and perf files
    for main_dir in dirs:
        list_dir = os.listdir(main_dir)
        path = ''
        for dir in list_dir:
            if dir == 'twitter' or dir == 'gigaword' or dir == 'word2vec':
                sub_dirs = os.listdir(f'{main_dir}/{dir}')
                for sub_dir in sub_dirs:
                    if os.path.isfile(f'{main_dir}/{dir}/{sub_dir}'):
                        continue
                    else:
                        path = f'{main_dir}/{dir}/{sub_dir}' 
                        update_names(path)
            else:
                path = f'{main_dir}/{dir}'
                update_names(path)
