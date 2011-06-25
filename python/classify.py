#!/usr/bin/python
#
# load a pre-trained classifier
#

import nltk.classify.util
from nltk.classify import maxent, NaiveBayesClassifier
import sys, pickle

def load_dataset(filename):
    f = open(filename,"rb") 
    c = pickle.load(f) 
    f.close() 
    return c

def main():
    c = load_dataset('naive_bigrams.dat')
    print c.classify('bonito carro')

if __name__ == "__main__":
    sys.exit(main())
