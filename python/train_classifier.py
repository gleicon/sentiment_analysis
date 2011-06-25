#!/usr/bin/python

# nltk classifiers
# given a tag and a folder, trains all listed classifiers and save their databases
# used classifiers:
# nltk.classify.NaiveBayesClassifier
# nltk.classify.maxent
# no parameters -> run movie demo

import nltk.classify.util
from nltk.classify import maxent, NaiveBayesClassifier
from filters import * 
import pickle
import sys, getopt


def save_dataset(filename, classifier):
    f = open(filename,"wb") 
    pickle.dump(classifier, f,1) 
    f.close() 

def load_dataset(filename):
    f = open(filename,"rb") 
    pickle.load(f) 
    f.close() 


def train_and_show_results(pos, neg, pos_bigrams, neg_bigrams, pos_control, neg_control, pos_control_bigrams, neg_control_bigrams):
    if pos_control == None or neg_control == None or pos_control_bigrams == None or neg_control_bigrams == None:
        negcutoff = len(neg)*3/4
        poscutoff = len(pos)*3/4
        neg_bigrams_cutoff = len(neg_bigrams)*3/4
        pos_bigrams_cutoff = len(pos_bigrams)*3/4
        test_bag_of_words = neg[negcutoff:] + pos[poscutoff:]
        test_bigrams = neg_bigrams[neg_bigrams_cutoff:] + pos_bigrams[pos_bigrams_cutoff:]
        train_corpora_bag_of_words = neg[:negcutoff] + pos[:poscutoff]
        train_corpora_bigrams = neg_bigrams[:neg_bigrams_cutoff] + pos_bigrams[:pos_bigrams_cutoff]
    else:
        test_bag_of_words = neg_control + pos_control
        test_bigrams = neg_control_bigrams + pos_control_bigrams
        train_corpora_bag_of_words = neg+pos
        train_corpora_bigrams = neg_bigrams + pos_bigrams
    
    print "negative corpus: ", len(neg) 
    print "positive corpus: ", len(pos)

    if neg_control != None:
        print "negative test corpus: ", len(neg_control) 
        print "positive test corpus: ", len(pos_control)

    print 'bag of words and bigrams - Naive Bayes' 
    naive_bayes = NaiveBayesClassifier.train(train_corpora_bag_of_words)
    naive_bayes_bigrams = NaiveBayesClassifier.train(train_corpora_bigrams)
   
    save_dataset('naive_bayes.dat', naive_bayes)
    save_dataset('naive_bayes_bigrams.dat', naive_bayes_bigrams)
    
    print 'bag of words and bigrams - Maximum Entropy' 
    maximum_entropy = nltk.MaxentClassifier.train(train_corpora_bag_of_words, max_iter=2)
    maximum_entropy_bigrams = nltk.MaxentClassifier.train(train_corpora_bigrams, max_iter=2)
    
    save_dataset('maximum_entropy.dat', maximum_entropy)
    save_dataset('maximum_entropy_bigrams.dat', maximum_entropy_bigrams)

    print 'Naive Bayesian results'
    print 'bag of words' 
    print 'Accuracy:', nltk.classify.util.accuracy(naive_bayes, test_bag_of_words)
    naive_bayes.show_most_informative_features()  
    print_precision_recall(naive_bayes, test_bag_of_words) 


    print '\nbigrams'
    print 'Accuracy:', nltk.classify.util.accuracy(naive_bayes_bigrams, test_bigrams)
    naive_bayes_bigrams.show_most_informative_features()  
    print_precision_recall(naive_bayes_bigrams, test_bigrams) 

    print 'Maximum Entropy results'
    print 'bag of words' 
    print 'Accuracy:', nltk.classify.util.accuracy(maximum_entropy, test_bag_of_words)
    maximum_entropy.show_most_informative_features()  
    print_precision_recall(maximum_entropy, test_bag_of_words) 


    print '\nbigrams'
    print 'Accuracy:', nltk.classify.util.accuracy(maximum_entropy_bigrams, test_bigrams)
    maximum_entropy_bigrams.show_most_informative_features()  
    print_precision_recall(maximum_entropy_bigrams, test_bigrams) 

def main():
    argv = sys.argv
    opts = args = None
    try:
        opts, args = getopt.getopt(argv[1:], "hlp:n:s:w:a:b:") # a: positive control, b: negative control
    except getopt.error, e:
        print "error: ", e
        print "usage %s -t tag -f folder [-s portuguese|english stemmer] [-w portuguese|english stopwords removal] [-l(ower case)]" % argv[0]
        sys.exit(-1)

    neg_control_folder=pos_control_folder=positive_folder = negative_folder = stemmer = stpwords = lower = None
    for option, value in opts:
        if option in ("-h", "--help"):
            raise Usage(help_message)
        if option in ("-s", "--stemmer"):
            stemmer = value
        if option in ("-w", "--stopwords"):
            stpwords = value
        if option in ("-l", "--lower"):
            lower = True
        if option in ("-p", "--positive_folder"):
            positive_folder = value
        if option in ("-n", "--negative_folder"):
            negative_folder = value
        if option in ("-a", "--positive_control_folder"):
            pos_control_folder = value
        if option in ("-b", "--negative_control_folder"):
            neg_control_folder = value
    pos_control = neg_control = pos_control_bigrams = neg_control_bigrams = None 
    if positive_folder == None or negative_folder == None:
        import demo_movie_reviews
        (pos, neg, pos_bigrams, neg_bigrams) = demo_movie_reviews.setup_demo(lower)
    else: 
        neg = create_corpus_from_dir_and_tag (negative_folder, "negative", stemmer, stpwords, lower)
        pos = create_corpus_from_dir_and_tag (positive_folder, "positive", stemmer, stpwords, lower)
        pos_bigrams = create_corpus_from_dir_and_tag(negative_folder, "negative", stemmer, stpwords, lower, wordlist_to_bigrams_dict) 
        neg_bigrams = create_corpus_from_dir_and_tag(positive_folder, "positive", stemmer, stpwords, lower, wordlist_to_bigrams_dict)         
        if pos_control_folder != None or neg_control_folder != None:
            pos_control = create_corpus_from_dir_and_tag(pos_control_folder, "positive", stemmer, stpwords, lower)
            neg_control = create_corpus_from_dir_and_tag(neg_control_folder, "negative", stemmer, stpwords, lower) 
            pos_control_bigrams = create_corpus_from_dir_and_tag(pos_control_folder, "positive", stemmer, stpwords, lower, wordlist_to_bigrams_dict)
            neg_control_bigrams = create_corpus_from_dir_and_tag(neg_control_folder, "negative", stemmer, stpwords, lower, wordlist_to_bigrams_dict) 

    train_and_show_results(pos, neg, pos_bigrams, neg_bigrams, pos_control, neg_control, pos_control_bigrams, neg_control_bigrams)

if __name__ == "__main__":
    sys.exit(main())
