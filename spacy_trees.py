import spacy
from nltk.tree import Tree

TRAIN_SOURCE = '../project1-train-dev/train.gold'
DEV_SOURCE = '../project1-train-dev/dev.gold'

def get_sents(filename):
    """Turn a gold file into lists of tuples for spacy dependency trees
    Inputs:
        filename: path to gold file
    Returns:
        sents: list of strings
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
                            #print(current_sent) #for debugging
                            new_sent = ' '.join(current_sent)
                            #print(new_sent) #for debugging
                            sents.append(new_sent)
                        #Make a tuple of everything but the index
                        #And add to current_sent
                        current_sent = [line_tokens[1]]
                    else:
                        current_sent.append(line_tokens[1])
    return sents

def get_trees(filename):
    with open(filename) as source:
        raw = source.read()
        nlp = spacy.load('en') #tokenizer, pos-tag, dependencies, ner
        doc = nlp(raw)
        return doc

def tok_format(tok):
    #https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node):
    #https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)

def print_tree(doc,k):
    #print tree number k
    sents = list(doc.sents)
    sent = sents[k]
    to_nltk_tree(sent.root).pretty_print()


if __name__ == '__main__':
    #print("Constructing sentences from gold file...")
    #sents = get_sents(TRAIN_SOURCE)

    print("Running sentnences through nlp pipeline...")
    doc = get_trees(TRAIN_SOURCE)

    print("Printing example tree...")
    for i in range(10):
        print_tree(doc,i)
