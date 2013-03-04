# Copyright (C) 2013 Wesley Baugh
from nose.tools import assert_raises

from infertweet.nlp import FeatureExtractor
from infertweet.classify import MultinomialNB


class TestFeatureExtractor(object):
    """Extract features from text for use in classification."""
    def setup(self):
        self.extractor = FeatureExtractor()
        self.document = 'I am so happy about this project'.split()

    def test_no_features(self):
        result = self.extractor.extract(self.document)
        assert result == []

    def test_ngrams(self):
        test = [(1, 1, ['I', 'am', 'so', 'happy', 'about', 'this', 'project']),
                (2, 2, ['__start__ I', 'I am', 'am so', 'so happy', 'happy about',
                        'about this', 'this project', 'project __end__']),
                (3, 3, ['__start__ I am', 'I am so', 'am so happy',
                        'so happy about', 'happy about this',
                        'about this project', 'this project __end__'])]
        test.append((1, 2, test[0][2] + test[1][2]))
        test.append((2, 3, test[1][2] + test[2][2]))
        test.append((1, 3, test[0][2] + test[1][2] + test[2][2]))
        failed = []
        for min_n, max_n, expected in test:
            self.extractor.min_n, self.extractor.max_n = min_n, max_n
            result = sorted(self.extractor.extract(self.document))
            # Convert expected to a tuple.
            expected = [tuple(x.split()) for x in expected]
            expected = sorted(expected)
            if result != expected:
                failed.append(((min_n, max_n), result, expected))

        # This explicitly shows the expected tuple return type.
        document = 'this is a test'.split()
        expected = [('__start__', 'this'), ('this', 'is'), ('is', 'a'),
                    ('a', 'test'), ('test', '__end__')]
        expected = sorted(expected)
        self.extractor.min_n, self.extractor.max_n = 2, 2
        result = sorted(self.extractor.extract(document))
        if result != expected:
            failed.append(((2, 2), result, expected))

        assert not failed

    def test_ngrams_multinomialnb(self):
        """Integration test with Naive Bayes classifier."""
        classifier = MultinomialNB()
        self.extractor.min_n, self.extractor.max_n = 1, 3
        features = self.extractor.extract(self.document)
        classifier.train([features, 'positive'])

    def test_ngrams_non_zero(self):
        bad_ranges = [(0, 1), (1, 0)]
        for min_n, max_n in bad_ranges:
            self.extractor.min_n, self.extractor.max_n = min_n, max_n
            assert_raises(ValueError, self.extractor.extract, self.document)

    def test_ngrams_non_negative(self):
        bad_ranges = [(-1, 1), (1, -1), (-2, -1), (-1, 0), (0, -1)]
        for min_n, max_n in bad_ranges:
            self.extractor.min_n, self.extractor.max_n = min_n, max_n
            assert_raises(ValueError, self.extractor.extract, self.document)

    def test_ngrams_non_reversed(self):
        bad_ranges = [(2, 1), (3, 1), (3, 2)]
        for min_n, max_n in bad_ranges:
            self.extractor.min_n, self.extractor.max_n = min_n, max_n
            assert_raises(ValueError, self.extractor.extract, self.document)
