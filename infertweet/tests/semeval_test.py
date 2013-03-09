# Copyright (C) 2013 Wesley Baugh
import ast

from unidecode import unidecode

from infertweet import semeval as s


EXAMPLE_DATA = """
264169034155696130\t382403760\t"neutral"\tNot Available
254941790757601280\t557103111\t"negative"\tThey may have a SuperBowl in Dallas, but Dallas ain't winning a SuperBowl. Not with that quarterback and owner. @S4NYC @RasmussenPoll
263939499426476032\t37877566\t"positive"\t@jackseymour11 I may have an Android phone by the time I get back to school! :)
263079812954992640\t392639346\t"neutral"\t#web Apple software, retail chiefs out in overhaul: SAN FRANCISCO (Reuters) - Apple Inc CEO Tim Cook on Monday replaced the heads of ...
264169073632505857\t191125886\t"objective-OR-neutral"\tIndiana 1, Northwestern 0, end first half, men's soccer. Eriq Zavaleta's 16th goal the difference. IU dominating play. #iusocc
212392538055778304\t274996324\t"objective"\tWhy is "Happy Valentines Day" trending? It's on the 14th of February not 12th of June smh..
NA\tDEV-200024\tpositive\tLana Del Rey at Hammersmith Apollo in May... Very badly want tickets
NA\tDEV-200025\tneutral\tShaw wouldn\u2019t let Luck throw late in the FIesta Bowl\u002c but he\u2019s fine with Nunes throwing a fade route on 4th and 4 w/ 1:50 left.
NA\tDEV-200026\tnegative\tMonday before I leave Singapore\u002c I am going to post something that might be offensive.
NA\tDEV-200001\tunknwn\tWon the match #getin . Plus\u002c tomorrow is a very busy day\u002c with Awareness Day\u2019s and debates. Gulp. Debates...
NA\tTEST-400003\tunknwn\tOn Radio786 100.4fm 7:10 Fri Oct 19 Labour analyst Shawn Hattingh: Cosatu\u2019s role in the context of unrest in the mining http://t.co/46pjzzl6
11051\tSMS-600002\tunknwn\tcan u tape the match for me?  i\u2019ll rush over straight after the dinner
"""
EXAMPLE_DATA = EXAMPLE_DATA.strip().split('\n')


class TestTaskBGenerator(object):
    def setup(self):
        self.tweets = s.task_b_generator(EXAMPLE_DATA)

    def test_labels(self):
        # Just a note: I ran `timeit`, and wrapping `labels` in a `set`
        # really does improve performance (2x), even when the size of a list
        # is this small.
        labels = set(['positive', 'negative', 'neutral', None])
        for tweet in self.tweets:
            assert tweet.label in labels

    def test_sid(self):
        for tweet in self.tweets:
            assert tweet.sid

    def test_uid(self):
        for tweet in self.tweets:
            assert tweet.uid

    def test_text_not_unavailable(self):
        for tweet in self.tweets:
            assert tweet.text != 'Not Available'

    def test_text_python_parsed(self):
        for tweet in self.tweets:
            result = tweet.text
            try:
                expected = ast.literal_eval(''.join(['u"', tweet.text, '"']))
            except SyntaxError:  # Wasn't written using `repr`.
                continue
            assert result == expected

    def test_text_not_unicode(self):
        for tweet in self.tweets:
            result = tweet.text
            expected = unidecode(tweet.text)
            assert result == expected

    def test_count(self):
        count = sum(1 for x in self.tweets)
        num_unavailable = sum(x.rsplit('\t', 1)[-1] == 'Not Available' for x
                              in EXAMPLE_DATA)
        expected = len(EXAMPLE_DATA) - num_unavailable
        assert count == expected
