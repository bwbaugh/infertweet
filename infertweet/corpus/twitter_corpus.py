# Copyright (C) 2013 Wesley Baugh
import bz2
import codecs
import json
from collections import namedtuple

# import langid


def tweet_generator(*corpora):
    Tweet = namedtuple('Tweet', 'text')

    def parse_lines(fobj):
        for line in fobj:
            try:
                data = json.loads(line)
            except ValueError:  # Possibly incomplete line in corpus file
                continue
            # if 'prep' not in corpus:
            #     lang, prob = langid.classify(status.text)
            #     if lang != 'en' or prob < 0.50:
            #         continue
            yield Tweet(data['text'].encode('utf-8'))

    for corpus in corpora:
        print 'Parsing file:', corpus
        try:
            with bz2.BZ2File(corpus, mode='r') as f:
                for tweet in parse_lines(f):
                    yield tweet
        except IOError:
            with codecs.open(corpus, encoding='utf-8') as f:
                for tweet in parse_lines(f):
                    yield tweet
