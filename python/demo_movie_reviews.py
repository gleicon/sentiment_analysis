# demo test - nltk movie reviews
from nltk.corpus import movie_reviews
from filters import create_corpus_from_file_list, wordlist_to_bigrams_dict 

nltk_movie_reviews_data_root = '/Users/gleicon/nltk_data/corpora/movie_reviews/'

def setup_demo(lower):
    print 'running movie reviews demo. data dir: ', nltk_movie_reviews_data_root
    negative_reviews = map (lambda x: nltk_movie_reviews_data_root + x, movie_reviews.fileids('neg'))
    positive_reviews = map (lambda x: nltk_movie_reviews_data_root + x, movie_reviews.fileids('pos'))
    pos = create_corpus_from_file_list(negative_reviews, "negative", None, None, lower) 
    neg = create_corpus_from_file_list(positive_reviews, "positive", None, None, lower)         
    pos_bigrams = create_corpus_from_file_list(negative_reviews, "negative", None, None, lower, wordlist_to_bigrams_dict) 
    neg_bigrams = create_corpus_from_file_list(positive_reviews, "positive", None, None, lower, wordlist_to_bigrams_dict)         
    return (pos, neg, pos_bigrams, neg_bigrams)
