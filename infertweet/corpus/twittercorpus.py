# Copyright (C) 2013 Wesley Baugh
"""Sander's Twitter Sentiment Corpus.

See: <http://www.sananalytics.com/lab/twitter-sentiment/>
"""
import codecs

from infer.experiment import Experiment

from infertweet.config import get_config


class TestSandersCorpus(Experiment):
    def _test_data(self):
        config = get_config()
        with codecs.open(config.get('sanders', 'corpus'),
                         encoding='utf-8') as f:
            for line in f:
                line = line.split(',', 5)
                if len(line) != 5:
                    continue
                sentiment = line[1][1:-1].encode('utf-8')
                if sentiment not in ['positive', 'negative', 'neutral']:
                    continue
                text = line[4].encode('utf-8')
                # Strip surrounding '"'
                text = text[1:-1]
                yield self.DataInstance(text, sentiment)
