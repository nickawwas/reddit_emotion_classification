import sys
import os
from Models import Models

def train_models(dataset, model_name, is_demo):
    splits = input('Enter a value for the training split (multiple can be entered) values: 0 < 1 ')
    
    train_test = []
    for split in splits.split():
        train_test.append("{:.2f}".format(float(split)))

    if is_demo:
        output_file = 'demo_performance.txt'
        model_name = 'demo'
    elif model_name.split('_')[0] == 'word2vec':
        output_file = 'performance_w2v.txt'
        model_name = 'models/word2vec'
    else:
        output_file = 'performance.txt'
        model_name = 'models'

    os.system(f'python create_models.py {dataset} {model_name} {output_file} {train_test}')
    return

def test_models():
    return

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Please run command following the format shown below:')
        print('"python run_models.py {model_name} {dataset} {type} {train or test}"')
        print('Options: ... TODO')
        exit()

    model_name = dataset = type = method = ''
    for i, arg in enumerate(sys.argv):
        if i == 1:
            model_name = str(arg)
        elif i == 2:
            dataset = str(arg)
        elif i == 3:
            type = str(arg)
        elif i == 4:
            method = str(arg)

    is_demo = input('Is this a demo? Y/N: ')
    if is_demo.lower() == 'y':
        is_demo = True
    else:
        is_demo = False

    if method == 'train':
        train_models(dataset, model_name, is_demo)
    elif method == 'test':
        test_models()
    else:
        print('Invalid method input, please use: train or test')
