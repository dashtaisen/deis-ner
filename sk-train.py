"""CRF training with scikit-learn API
"""

from itertools import chain

import nltk
import sklearn
import scipy.stats
import re
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from nltk.corpus import wordnet as wn

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

#http://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html


#Location of training, dev, and test sets
TRAIN_SOURCE = '../train.gold'
DEV_SOURCE = '../dev.gold'
TEST_SOURCE = '../dev.gold'
EVALUATE_OUTPUT = False
OUTPUT_FILE = 'sk-train.model'

def get_tuples(filename):
    """Turn a gold file into lists of tuples for CRF processing
    Inputs:
        filename: path to gold file
    Returns:
        sents: list of list of tuples
    """
    sents = list()

    #TODO: do this with less nesting
    with open(filename) as source:
        #The current sentence: a list of (token, pos, label) tuples
        current_sent = list()

        for line in source:
            #If it's not blank. Possibly redundant with line below.
            if len(line) > 0:
                #Split at whitespace. Result will be [index, token, pos, label]
                line_tokens = line.split()
                #print(line_tokens) #for debugging
                if len(line_tokens) > 0: #If not a blank line
                    if line_tokens[0] == '0': #index==0 means start of new sentence
                        if len(current_sent) > 0: #add the current sentence to sents
                            sents.append(current_sent)
                        #Make a tuple of everything but the index
                        #And add to current_sent
                        current_sent = [tuple(line_tokens[1:])]
                    else:
                        current_sent.append(tuple(line_tokens[1:]))
    return sents

def word2features(sent, i):
    """Convert sentence to features.
    Inputs:
        sent: list of (word, pos, label) tuples
        i: index of particular (word,pos,label) in the sent
    Returns:
        features: dict of features
    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word[-5:]': word[-5:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'capPeriod': word.istitle() and word[-1] == '.',
        #'oneCap': len(word) == 1 and word.isupper(),
        #'allCapPeriod': word[:-1].isupper() and word[-1] == '.',
        #'hyphen': '-' in word, #from http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
        #'word[:2]': word[:2],
        #'word[:3]': word[:3],
        #'word[:4]': word[:4],
        #'word[:5]': word[:5],
        'pattern': getPattern(word),
        'patternSumm': getPattern(word,True),
        'postag': postag,
        'postag[:2]': postag[:2],
        'wordnet-neg':len(wn.synsets(word)) == 0
    }
    
    #from http://www.oegai.at/konvens2012/proceedings/17_tkachenko12o/17_tkachenko12o.pdf
    '''
    if '-' in word: 
        subtokenI = 0
        for subtoken in word.split('-'):
            features['subtoken'+str(subtokenI)] = subtoken
            subtokenI += 1
    '''
    
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            #'-1:pattern': getPattern(word1),
            '-1:ampersand': word1 == '&',
            '-1:patternSumm': getPattern(word1,True),
            '-1:postag[:2]': postag1[:2],
            '-1:wordnet-neg':len(wn.synsets(word1)) == 0
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
            #'+1:pattern': getPattern(word1),
            '+1:ampersand': word1 == '&',
            '+1:patternSumm': getPattern(word1,True),
            '+1:postag[:2]': postag1[:2],
            '+1:wordnet-neg':len(wn.synsets(word1)) == 0
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    """Convert entire sentence to features
    Inputs:
        sent: list of (word, pos, label) tuples
    Returns:
        list of feature dicts
    """
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    """Extract labels from sentence
    Inputs:
        sent: list of (word, pos, label) tuples
    Returns:
        list of labels
    """
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    """Extract tokens from sentence
    Inputs:
        sent
    Returns:
        list of tokens
    """
    return [token for token, postag, label in sent]

def train(X_train, y_train):
    """Train CRF model
    Inputs:
        X_train: list of feature dicts for training set
        y_train: list of labels for training set
    Returns:
        model: trained CRF model
    """
    model = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(crf, X_dev, y_dev):
    """Evaluate the model
    Inputs:
        crf: trained CRF model
        X_dev: list of feature dicts for dev set
        y_dev: list of labels for dev set
    Returns:
        None (prints metrics)
    """

    #Get the labels we're evaluating
    labels = list(crf.classes_)

    #Most labels are 'O', so we ignore,
    #otherwise our scores will seem higher than they actually are.
    labels.remove('O')

    print("Predicting labels")
    y_pred = crf.predict(X_dev)

    #print(y_pred[:10]) #for debugging
    #print(y_dev[:10]) #for debugging

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
    
def getPattern(word,summarized=False):
    """Return the word with all capitals as A, all lowercase as a, all numbers as 0, and all punctuation as -
    Inputs:
        word: The word to convert to a pattern
        summarized: whether to collapse sequences of the same type of character
    Returns:
        A pattern string modeled on the input word
    """
    
    if summarized:
        pattern = re.sub(r'[A-Z]+','A',word)
        pattern = re.sub(r'[a-z]+','a',pattern)
        pattern = re.sub(r'[0-9]+','0',pattern)
        pattern = re.sub(r'[^A-Za-z0-9]+','-',pattern)
    else:
        pattern = re.sub(r'[A-Z]','A',word)
        pattern = re.sub(r'[a-z]','a',pattern)
        pattern = re.sub(r'[0-9]','0',pattern)
        pattern = re.sub(r'[^A-Za-z0-9]','-',pattern)
    
    return pattern
    
def output_model(crf,x_dev):
    """Print predicted tags to file, one line per tag with a blank line in between sentences
    Inputs:
        crf: Trained CRF model
        x_dev: List of lists of feature dictionaries of test set
    """
    
    with open(OUTPUT_FILE,'w') as f:
        #Get the labels we're evaluating
        labels = list(crf.classes_)

        #Most labels are 'O', so we ignore,
        #otherwise our scores will seem higher than they actually are.
        labels.remove('O')
        
        y_pred = crf.predict(x_dev)
        
        for sentence in y_pred:
            for label in sentence:
                f.write(label)
                f.write('\n')
            f.write('\n')
                

if __name__ == "__main__":
    print("Converting gold files to sentence tuples...")
    train_sents = get_tuples(TRAIN_SOURCE)
    dev_sents = get_tuples(DEV_SOURCE)

    #print(train_sents[2]) #for debugging
    #print(sent2features(train_sents[0])[0]) #for debugging

    print("Building training and dev sets...")
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_dev = [sent2features(s) for s in dev_sents]
    y_dev = [sent2labels(s) for s in dev_sents]

    print("Training model...")
    crf = train(X_train, y_train)

    if EVALUATE_OUTPUT:
        print("Evaluating model...")
        evaluate_model(crf, X_dev[:1], y_dev[:1])
    else:
        output_model(crf,X_dev,y_dev)
        
