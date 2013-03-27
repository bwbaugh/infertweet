# Copyright (C) 2013 Wesley Baugh
"""Configuration file access and settings."""
import errno
import os
import sys
import ConfigParser


CONFIG_FNAME = 'infertweet.ini'


def create_default_config():
    """Create a default config file."""
    config = ConfigParser.SafeConfigParser()

    config.add_section('art_festival')
    config.set('art_festival', 'corpus', 'PATH/art_tweets_all_sentiment.tsv')

    config.add_section('sanders')
    config.set('sanders', 'corpus', 'PATH/test-corpus.csv')

    config.add_section('semeval')
    config.set('semeval', 'training', 'PATH/tweeti-b.dist.tsv.data')
    config.set('semeval', 'development', 'PATH/twitter-dev-gold-B.tsv')
    config.set('semeval', 'twitter_test', 'PATH/twitter-test-input-B.tsv')
    config.set('semeval', 'sms_test', 'PATH/sms-test-input-B.tsv')
    config.set('semeval', 'twitter_predict',
               'PATH/task2-bwbaugh-B-twitter-unconstrained.output')
    config.set('semeval', 'sms_predict',
               'PATH/task2-bwbaugh-B-SMS-unconstrained.output')

    config.add_section('sentiment')
    config.set('sentiment', 'path', 'PATH/TO/CLASSIFIER/')
    config.set('sentiment', 'classifier', 'sentiment-classifier.pickle')
    config.set('sentiment', 'rpc_host', 'localhost')
    config.set('sentiment', 'rpc_port', '18861')
    config.set('sentiment', 'web_query_log', 'web_log_queries.txt')
    config.set('sentiment', 'chunk_size', '5000')
    config.set('sentiment', 'first_chunk', '100')
    config.set('sentiment', 'titles',
               '("Single Classifier", "Hierarchical Classifier")')
    config.set('sentiment', 'labels',
               '("SemEval", "Positive", "Negative", "Neutral", "Accuracy")')
    config.set('sentiment', 'test_scale', '1')

    config.add_section('stanford')
    config.set('stanford', 'corpus',
               'PATH/training.1600000.processed.noemoticon.shuffled.csv')

    config.add_section('twitter_corpus')
    config.set('twitter_corpus', 'emoticons', 'PATH/twitter-sentiment.json.bz2')

    config.add_section('web')
    config.set('web', 'port', '8080')
    config.set('web', 'gzip', 'true')
    config.set('web', 'debug', 'true')

    return config


def get_config(fname=CONFIG_FNAME, create=True, exit=True):
    """Reads a configuration file from disk."""
    config = ConfigParser.SafeConfigParser()
    try:
        with open(fname) as f:
            config.readfp(f)  # pragma: no branch
    except IOError as e:
        if e.errno != errno.ENOENT:
            raise  # pragma: no cover
        if create:
            print 'Configuration file not found! Creating one...'
            config = create_default_config()
            with open(fname, mode='w') as f:
                config.write(f)
            message = 'Please edit the config file named "{}" in directory "{}"'
        else:
            message = 'Configuration file "{}" not found in directory "{}"'
        print message.format(CONFIG_FNAME, os.getcwd())
        if exit:
            sys.exit(errno.ENOENT)
        else:
            if not create:
                return None
    return config
