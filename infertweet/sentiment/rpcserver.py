# Copyright (C) 2013 Brian Wesley Baugh
"""RPC server for serving the sentiment classifier."""
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import rpyc
from rpyc.utils.server import ThreadedServer

from infertweet.config import get_config

# Load requirements of the pickled classifier.
from collections import namedtuple
from infertweet.sentiment.tokenizer import tokenizer


def main():
    config = get_config()
    path = config.get('sentiment', 'path')
    sentiment_fname = config.get('sentiment', 'classifier')
    sentiment_classifier_location = os.path.join(path, sentiment_fname)

    print 'Loading classifier (may take a few minutes) ... ',
    assert namedtuple and tokenizer
    with open(sentiment_classifier_location, mode='rb') as f:
        sentiment_classifier = pickle.load(f)
    print 'DONE'

    extractor = sentiment_classifier[0]
    subjective = sentiment_classifier[1]
    polarity = sentiment_classifier[2]

    def train(*documents):
        for document, label in documents:
            if label != 'neutral':
                assert label in set(['positive', 'negative'])
                polarity.train((document, label))
                label = 'subjective'
            assert label in set(['neutral', 'subjective'])
            subjective.train((document, label))

    print 'Training (pickled) classifier using misclassified instances...',
    with open(config.get('web', 'misclassified_file')) as f:
        for line in f:
            date, user, flag, mislabel, text = line.rstrip().split('\t')
            features = extractor.extract(text)
            train((features, flag))
    print 'DONE'

    class SentimentService(rpyc.Service):
        def exposed_extract(self, document):
            return extractor.extract(document)

        def exposed_subjective_classify(self, features):
            return subjective.classify(features)

        def exposed_subjective_conditional(self, feature, label):
            return subjective.conditional(feature, label)

        def exposed_polarity_classify(self, features):
            return polarity.classify(features)

        def exposed_polarity_conditional(self, feature, label):
            return polarity.conditional(feature, label)

    rpc_port = int(config.get('sentiment', 'rpc_port'))
    t = ThreadedServer(SentimentService, port=rpc_port)
    t.start()


if __name__ == '__main__':
    main()
