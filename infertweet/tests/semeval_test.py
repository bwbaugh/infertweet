# Copyright (C) 2013 Wesley Baugh
from infertweet import semeval as s


EXAMPLE_DATA = """
264169034155696130\t382403760\t"neutral"\tNot Available
254941790757601280\t557103111\t"negative"\tThey may have a SuperBowl in Dallas, but Dallas ain't winning a SuperBowl. Not with that quarterback and owner. @S4NYC @RasmussenPoll
263939499426476032\t37877566\t"positive"\t@jackseymour11 I may have an Android phone by the time I get back to school! :)
263079812954992640\t392639346\t"neutral"\t#web Apple software, retail chiefs out in overhaul: SAN FRANCISCO (Reuters) - Apple Inc CEO Tim Cook on Monday replaced the heads of ...
264169073632505857\t191125886\t"objective-OR-neutral"\tIndiana 1, Northwestern 0, end first half, men's soccer. Eriq Zavaleta's 16th goal the difference. IU dominating play. #iusocc
"""
EXAMPLE_DATA = EXAMPLE_DATA.strip().split('\n')


class TestTaskBGenerator(object):
    def setup(self):
        self.tweets = s.task_b_generator(EXAMPLE_DATA)

    def test_labels(self):
        # Just a note: I ran `timeit`, and wrapping `labels` in a `set`
        # really does improve performance (2x), even when the size of a list
        # is this small.
        labels = set(['positive', 'negative', 'neutral'])
        for tweet in self.tweets:
            assert tweet.label in labels

    def test_sid(self):
        for tweet in self.tweets:
            int(tweet.sid)

    def test_uid(self):
        for tweet in self.tweets:
            int(tweet.uid)

    def test_text_not_unavailable(self):
        for tweet in self.tweets:
            assert tweet.text != 'Not Available'

    def test_count(self):
        count = sum(1 for x in self.tweets)
        num_unavailable = sum(x.rsplit('\t', 1)[-1] == 'Not Available' for x
                              in EXAMPLE_DATA)
        expected = len(EXAMPLE_DATA) - num_unavailable
        assert count == expected
