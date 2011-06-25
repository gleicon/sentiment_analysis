require 'hoatzin'

#if File.exists? 'sentiment_svm_metadata' and File.exists? 'sentiment_svm_model' 
#  c = Hoatzin::Classifier.new(:metadata => 'sentiment_svm_metadata', :model => 'sentiment_svm_model')
#else
c = Hoatzin::Classifier.new()
#end

class SvmClassifier 
  def initialize
    @c = Hoatzin::Classifier.new()
  end
  def create_corpus(dirname)
    return Dir.entries(dirname).map { |f| "#{dirname}/#{f}" if not File.directory? f }.compact
  end

  def train(tag, corpus)
    corpus.each do |doc|
      @c.train(tag, File.read(doc))
    end
  end

  def classify_and_count(corpus)
    res = Hash.new(0)
    corpus.each do |doc|
      res[@c.classify(File.read(doc))] +=1
    end
    res
  end
  
  def test
    positive_corpus = create_corpus 'positive'
    negative_corpus = create_corpus 'negative'
    positive_test_corpus = create_corpus 'test_positive'
    negative_test_corpus = create_corpus 'test_negative'

    train :positive, positive_corpus
    train :negative, negative_corpus

    @c.sync

    @c.save(:metadata => 'sentiment_svm_metadata', :model => 'sentiment_svm_model')

    puts 'testing for positive'
    puts classify_and_count positive_test_corpus

    puts 'testing for negative'
    puts classify_and_count negative_test_corpus    
  end

end

sc = SvmClassifier.new
sc.test



