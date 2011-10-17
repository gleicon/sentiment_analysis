# filters and file handling module for nltk classify 
# there is code inspired by steamhacker.com

from nltk.stem import RSLPStemmer, PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import precision, recall, f_measure
import codecs
import os, glob
import itertools

# stemmer and stopwords table
g_stemmers = defaultdict(lambda: None) 
g_stopwords = defaultdict(lambda: None)

g_stemmers.update({"portuguese": RSLPStemmer(), "english":PorterStemmer()})
g_stopwords.update({"portuguese": stopwords.words('portuguese'), 
            "english": stopwords.words('english')})

# feature extraction filters
# bag of words
def wordlist_to_dict(words):
    return defaultdict(lambda: None, [(word, True) for word in words])

# bigrams
def wordlist_to_bigrams_dict(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return defaultdict(lambda: None, [(ngram, True) for ngram in itertools.chain(words, bigrams)])

# IR filters
def check_stopwords(data, sw):
    ret = []
    for a in data:
        if a.lower().encode('utf-8') not in g_stopwords[sw]:
           ret.append(a)
    return " ".join(ret)

def stemo(data, st):
    ret = []
    for a in data:
        ret.append(g_stemmers[st].stem(a))
    return ret

def lower_case(b):
    r=[]
    for w in b:
        r.append(w.lower())
    return r

# buffer handling
def create_corpus_from_file_list(allfiles, tag, stemmer, stpwords, lower, feat_extractor=wordlist_to_dict):
    if stopwords: sw = g_stopwords[stpwords]
    if stemmer: st = g_stemmers[stemmer]
    corpus = []
    for infile in allfiles:
        n = apply_filters_to_file(tag, infile, stemmer, stpwords, lower)
        corpus.append((feat_extractor(n), tag))
    return corpus 

# file handling
def create_corpus_from_dir_and_tag(folder, tag, stemmer, stpwords, lower, feat_extractor=wordlist_to_dict):
    allfiles = glob.glob(os.path.join(folder, '*.txt'))
    return create_corpus_from_file_list(allfiles, tag, stemmer, stpwords, lower, feat_extractor)
    
def apply_filters_to_file(tag, infile, st, sw, l):
    body = get_body(infile)
    if body == None:
        print 'empty file: ', infile
        return
    if l != None:
        body = lower_case(body)
    if sw != None:
        body = check_stopwords(body, sw)
    if st != None:
        body = stemo(body, st)
    return body

def get_body(filename):
    try:
        f = codecs.open(filename, "r", "utf-8" )
        b = f.read()
        f.close()
        return b.split()
    except IOError:
        print "File not found"
        return None

def print_precision_recall(classifier, test_dict):
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    for i, (feats, label) in enumerate(test_dict):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    print 'pos precision:', precision(refsets['positive'], testsets['positive'])
    print 'pos recall:', recall(refsets['positive'], testsets['positive'])
    print 'pos F-measure:', f_measure(refsets['positive'], testsets['positive'])
    print 'neg precision:', precision(refsets['negative'], testsets['negative'])
    print 'neg recall:', recall(refsets['negative'], testsets['negative'])
    print 'neg F-measure:', f_measure(refsets['negative'], testsets['negative'])

