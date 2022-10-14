Repository Link: https://github.com/nickawwas/reddit_emotion_classification

COMP 472 - Mini-Project 1

# Setup guide:

## Setting up virtual environment
Ensure you Python version is either 3.7 or 3.8 and 64-bit

### For macOS/linux:

```
pip3 install venv
python -m venv comp472
source comp472/bin/activate
pip3 install -r requirements.txt
```

### For Windows:

```
pip install venv
python -m venv comp472
comp472/Scripts/activate
pip install -r requirements.txt
```

## Running program to create models
> Note this will take several hours to days to run
Run the command below:
```
python create_models.py
```

This will create the charts associated with the dataset (barchart and piechart)

A folder called `models` will be created, inside each of the classification methods 
will have their models stored, for both emotions and sentiments. These can be used
later to load, avoiding the need to rebuild the models


## Running program with pre-built models

Run the command below:
```
python pre_built.py {model_name} {type}
```

List of entries for `model_name`:
- decision_tree
- naive_bayes
- perceptron
- GridSearch_DCT
- GridSearch_MNB
- GridSearch_Perceptron

List of entries for `type`:
- emotions
- sentiments






