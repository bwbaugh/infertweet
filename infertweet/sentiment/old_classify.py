# Copyright (C) 2012 Brian Wesley Baugh
import copy
import re
import string
from collections import defaultdict

import nltk
import nltk.classify.util
from nltk.probability import FreqDist, ELEProbDist
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from unidecode import unidecode

# import corpus


tokenizer = nltk.tokenize.TreebankWordTokenizer()
lemmatizer = nltk.WordNetLemmatizer()


STOPWORDS = stopwords.words()
STOPWORDS = None
NEGATE = "not n't".split(' ')

# POS_EMOTICONS = (
#              ">:] :-) :) :o) :] :3 :c) :> =] 8) =) :} :^) "
#              ">:D :-D :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 8-D B^D "
#              ":-)) "
#              ":'-) :') "
#              ">;] ;-) ;) *-) *) ;-] ;] ;D ;^) "
#              ">:P :-P :P X-P x-p xp XP :-p :p =p :-b :b "
#              "<3 "
#              )
# NEG_EMOTICONS = (
#              ">:[ :-( :( :-c :c :-< :< :-[ :[ :{ "
#              ":'-( :'( "
#              "QQ "
#              )
POS_EMOTICONS = (
             ":-) :) =) "
             )
NEG_EMOTICONS = (
             ":-( :( "
             )
POS_EMOTICONS = POS_EMOTICONS.strip().split(' ')
# POS_EMOTICONS.extend([x[::-1] for x in POS_EMOTICONS])
NEG_EMOTICONS = NEG_EMOTICONS.strip().split(' ')
# NEG_EMOTICONS.extend([x[::-1] for x in NEG_EMOTICONS])

EMOTICONS = POS_EMOTICONS + NEG_EMOTICONS


class IncrementalNaiveBayes(object):
    """Builds the NB model externally, allowing incremental training.

    The source for this class is taken from the NLTK NaiveBayesClassifier
    class. Specifically, the train() method is split up so that training
    can be done incrementally instead of forcing it to be done all at once.
    """
    def __init__(self):
        self.label_freqdist = FreqDist()
        self.feature_freqdist = defaultdict(FreqDist)
        self.feature_values = defaultdict(set)
        self.fnames = set()

    def train(self, labeled_featuresets):
        """Incrementally train the NB classifier.
        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        """
        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            self.label_freqdist.inc(label)
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                self.feature_freqdist[label, fname].inc(fval)
                # Record that fname can take the value fval.
                self.feature_values[fname].add(fval)
                # Keep a list of all feature names.
                self.fnames.add(fname)

    def get_model(self, estimator=ELEProbDist):
        """Potentially unsafe to call more than a single time (but maybe OK)"""
        # Make a copy of the model to generate the classifier
        label_freqdist = self.label_freqdist
        feature_freqdist = copy.copy(self.feature_freqdist)
        feature_values = copy.copy(self.feature_values)
        fnames = self.fnames

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples - count)
                feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)


def filter_text(text):
    # Remove usernames
    text = re.sub(r'@([A-Za-z0-9_]+)', '', text)
    # Remove URLs
    # Source: http://daringfireball.net/2010/07/improved_regex_for_matching_urls
    text = re.sub(r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.]"""
                  r"""[a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+"""
                  r"""\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"""
                  r"""[^\s`!()\[\]{};:'".,<>?]))""", '', text)
    # Strip whitespace
    text = text.strip()
    return text


def regularlize_text(text):
    text = str(text)
    text = unidecode(text)
    text = text.lower()
    return text


def regularlize_tokens(tokens):
    tokens = list(tokens)
    previous_was_not = False
    for i, token in enumerate(tokens):
        # Shorten repeated chars (haaaaaaate -> haate)
        token = re.sub(r"(\w)\1{2,}", r'\1\1', token)
        # Lemmatize (women -> woman)
        token = lemmatizer.lemmatize(token)
        if previous_was_not and token not in NEGATE:
            token = 'not_' + token
        previous_was_not = token in NEGATE
        # Stopword and punctuation removal
        if (STOPWORDS and token in STOPWORDS) or token in string.punctuation:
            token = None
        # Done; update value in list
        tokens[i] = token
    tokens = [x for x in tokens if x]
    return tokens


def remove_emoticons(text):
    text = str(text)
    for emoticon in EMOTICONS:
        text = text.replace(emoticon, '')
    return text


def boolean_ngram_features(document, min_n=1, max_n=1):
    # Inspiration: http://stackoverflow.com/questions/7591258/fast-n-gram-calculation
    ngrams = dict()
    n_tokens = len(document)
    ngram_key = '{}_gram({})'
    for i in xrange(n_tokens):
        for j in xrange(i + min_n, min(n_tokens, i + max_n) + 1):
            ngrams[ngram_key.format(j - i, '_'.join(document[i:j]))] = True
    return ngrams


def boolean_pos_features(document):
    tagged = nltk.pos_tag(document)
    return dict([('pos(%s)' % pos, True) for word, pos in tagged])


def extract_features(text):
    features = dict()
    tokens = tokenizer.tokenize(text)
    # features.update(boolean_pos_features(tokens))
    tokens = regularlize_tokens(tokens)
    features.update(boolean_ngram_features(tokens, min_n=1, max_n=2))
    # features.update(boolean_ngram_features(tokens, min_n=3, max_n=3))
    return features


def get_noisy_sentiment(text):
    sentiment = None
    for emoticon in POS_EMOTICONS:
        if emoticon in text:
            sentiment = 'pos'
            break
    for emoticon in NEG_EMOTICONS:
        if emoticon in text:
            sentiment = None if sentiment else 'neg'
            break
    return sentiment


def _train(inc_nb, pos_instances, neg_instances, test, cutoff):
    print 'len(pos): {}, len(neg): {}'.format(len(pos_instances),
                                              len(neg_instances))

    pos_cutoff = int(len(pos_instances) * cutoff)
    neg_cutoff = int(len(neg_instances) * cutoff)

    train_feats = neg_instances[:neg_cutoff] + pos_instances[:pos_cutoff]
    test_feats = neg_instances[neg_cutoff:] + pos_instances[pos_cutoff:]
    print 'train on %d instances, test on %d instances' % (len(train_feats),
                                                           len(test_feats))
    inc_nb.train(train_feats)
    classifier = inc_nb.get_model()
    # classifier = NaiveBayesClassifier.train(trainfeats)
    if test:
        accuracy = nltk.classify.util.accuracy(classifier, test_feats)
        num_test = len(test_feats)
        print 'accuracy:', accuracy
        # classifier.show_most_informative_features(n=100)
    else:
        accuracy = None
        num_test = None
    inc_nb.train(test_feats)
    classifier = inc_nb.get_model()

    return classifier, accuracy, num_test


def overall_accuracy(accuracy):
    return sum(a * w for a, w in accuracy) / sum(w for a, w in accuracy)


def train_model(corpora, test=False, cutoff=0.75, chunk_size=20000,
                balanced=False):
    # Demo code based on the following source:
    #   http://streamhacker.com/2010/05/10/
    #       text-classification-sentiment-analysis-naive-bayes-classifier/
    if not test:
        cutoff = 1

    inc_nb = IncrementalNaiveBayes()
    classifier = None
    accuracy = []

    pos_instances = []
    neg_instances = []
    tweets = corpus.tweet_generator(*corpora)
    for tweet in tweets:
        tweet.text = filter_text(tweet.text)

        if 'RT' in tweet.text:
            continue

        tweet.text = regularlize_text(tweet.text)

        sentiment = get_noisy_sentiment(tweet.text)
        if not sentiment:
            continue

        tweet.text = remove_emoticons(tweet.text)
        features = extract_features(tweet.text)

        if sentiment == 'pos':
            pos_instances.append((features, sentiment))
        elif sentiment == 'neg':
            neg_instances.append((features, sentiment))

        if balanced:
            smallest = min(len(pos_instances), len(neg_instances))
        else:
            smallest = len(pos_instances) + len(neg_instances)
        if smallest == chunk_size:
            classifier = _train(inc_nb, pos_instances[:smallest],
                                neg_instances[:smallest], test, cutoff)
            classifier, inc_accuracy, num_test = classifier
            if test:
                accuracy.append((inc_accuracy, num_test))
                print 'current accuracy:', overall_accuracy(accuracy)
            pos_instances = pos_instances[smallest:]
            neg_instances = neg_instances[smallest:]
            yield classifier, inc_nb

    # Flush training buffer
    if pos_instances and neg_instances:
        smallest = min(len(pos_instances), len(neg_instances))
        if balanced:
            classifier = _train(inc_nb, pos_instances[:smallest],
                                neg_instances[:smallest], test, cutoff)
        else:
            classifier = _train(inc_nb, pos_instances,
                                neg_instances, test, cutoff)
        classifier, inc_accuracy, num_test = classifier
        if test:
            accuracy.append((inc_accuracy, num_test))

    if test:
        print 'overall accuracy:', overall_accuracy(accuracy)

    yield classifier, inc_nb
