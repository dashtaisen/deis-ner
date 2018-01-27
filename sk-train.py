"""CRF training with scikit-learn API
"""

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from nltk.corpus import wordnet as wn

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

#http://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

TRAIN_SOURCE = './train.gold'
DEV_SOURCE = './dev.gold'

def get_tuples(training_source):

    training_sents = list()

    with open(training_source) as source:
        current_sent = list()
        for line in source:
            if len(line) > 0:
                line_tokens = line.split()
                #print(line_tokens)
                if len(line_tokens) > 0:
                    if line_tokens[0] == '0':
                        if len(current_sent) > 0:
                            training_sents.append(current_sent)
                        current_sent = [tuple(line_tokens[1:])]
                    else:
                        current_sent.append(tuple(line_tokens[1:]))
    return training_sents

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'wordnet':len(wn.synsets(word)) > 0
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def train(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

def evaluate_model(crf, X_dev, y_dev):
    labels = list(crf.classes_)
    labels.remove('O')
    print("Predicting labels")
    y_pred = crf.predict(X_dev)
    print(y_pred[:10])
    print(y_dev[:10])
    print("Displaying accuracy")
    metrics.flat_f1_score(y_dev, y_pred,
                      average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print("Displaying detailed metrics")
    print(metrics.flat_classification_report(
        y_dev, y_pred, labels=sorted_labels, digits=3
    ))

if __name__ == "__main__":
    train_sents = get_tuples(TRAIN_SOURCE)
    dev_sents = get_tuples(DEV_SOURCE)
    print(train_sents[2])
    print(sent2features(train_sents[0])[0])
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    print("Training model")
    crf = train(X_train, y_train)

    X_dev = [sent2features(s) for s in dev_sents]
    y_dev = [sent2labels(s) for s in dev_sents]

    evaluate_model(crf, X_dev, y_dev)
