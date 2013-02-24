# Copyright (C) 2013 Wesley Baugh
from fractions import Fraction
from nose.tools import assert_raises

from infertweet.naive_bayes import MultinomialNB


class TestMultinomialNB(object):
    # This test uses the examples provided by:
    # http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
    def setup(self):
        self.training_docs = [
                              ('Chinese Bejing Chinese', 'yes'),
                              ('Chinese Chinese Shanghai', 'yes'),
                              ('Chinese Macao', 'yes'),
                              ('Tokyo Japan Chinese', 'no'),
                             ]
        self.training_docs = [(x.split(), y) for x, y in self.training_docs]
        self.classifier = MultinomialNB(*self.training_docs)

    def test_init_no_training(self):
        classifier = MultinomialNB()
        assert classifier.vocabulary == set()
        assert classifier.labels == set()
        classifier.train(*self.training_docs)
        self.test_labels()
        self.test_vocabulary()

    def test_train_one_document(self):
        documents = (['one', 'document', 'already', 'tokenized'], 'label')
        classifier = MultinomialNB(documents)
        expected = set(['one', 'document', 'already', 'tokenized'])
        assert classifier.vocabulary == expected

    def test_train_many_document(self):
        documents = [(['one', 'document', 'already', 'tokenized'], 'label')] * 5
        classifier = MultinomialNB(*documents)
        expected = set(['one', 'document', 'already', 'tokenized'])
        assert classifier.vocabulary == expected

    def test_train_not_tokenized(self):
        # Nothing should change if there is an error with the training docs.
        orig_label_count = self.classifier.label_count.copy()
        orig_label_vocab = self.classifier.label_vocab.copy()
        orig_label_token_count = self.classifier.label_token_count.copy()
        orig_label_length = self.classifier.label_length.copy()

        document = ('one document not tokenized', 'label')
        assert_raises(TypeError, self.classifier.train, document)

        assert orig_label_count == self.classifier.label_count
        assert orig_label_vocab == self.classifier.label_vocab
        assert orig_label_token_count == self.classifier.label_token_count
        assert orig_label_length == self.classifier.label_length

    def test_labels(self):
        expected = set(['yes', 'no'])
        assert self.classifier.labels == expected

    def test_vocabulary(self):
        expected = set(['Chinese', 'Bejing', 'Shanghai', 'Macao', 'Tokyo',
                        'Japan'])
        assert self.classifier.vocabulary == expected

    def test_label_token_count(self):
        tests = [
                 ('yes', 'Chinese', 5),
                 ('yes', 'Japan', 0),
                 ('no', 'Chinese', 1),
                 ('no', 'Japan', 1),
                ]
        for label, token, count in tests:
            assert self.classifier.label_token_count[label][token] == count

    def test_prior(self):
        tests = [
                 ('yes', Fraction(3, 4)),
                 ('no', Fraction(1, 4)),
                ]
        for label, prob in tests:
            assert self.classifier.prior(label) == prob

    def test_conditional(self):
        tests = [
                  ('Chinese', 'yes', Fraction(6, 14)),
                  ('Japan', 'yes', Fraction(1, 14)),
                  ('Chinese', 'no', Fraction(2, 9)),
                  ('Tokyo', 'no', Fraction(2, 9)),
                  ('Japan', 'no', Fraction(2, 9)),
                ]
        for token, label, prob in tests:
            assert self.classifier.conditional(token, label) == prob

    def test_score(self):
        tests = [
                 ('Chinese Chinese Chinese Tokyo Japan', 'yes',
                  Fraction(3, 4) * Fraction(3, 7) * Fraction(3, 7) *
                  Fraction(3, 7) * Fraction(1, 14) * Fraction(1, 14)),
                 ('Chinese Chinese Chinese Tokyo Japan', 'no',
                  Fraction(1, 4) * Fraction(2, 9) * Fraction(2, 9) *
                  Fraction(2, 9) * Fraction(2, 9) * Fraction(2, 9)),
                ]
        for document, label, score in tests:
            result = self.classifier.score(document.split(), label)
            assert result == score

    def test_score_not_tokenized(self):
        document, label = 'Chinese Chinese Chinese Tokyo Japan', 'yes'
        assert_raises(TypeError, self.classifier.score, document, label)

    def test_prob(self):
        tests = [
                 ('Chinese Chinese Chinese Tokyo Japan', 'yes',
                  Fraction(4782969, 6934265)),
                 ('Chinese Chinese Chinese Tokyo Japan', 'no',
                  Fraction(2151296, 6934265)),
                ]
        for document, label, prob in tests:
            result = self.classifier.prob(document.split(), label)
            assert result == prob

    def test_prob_not_tokenized(self):
        document, label = 'Chinese Chinese Chinese Tokyo Japan', 'yes'
        assert_raises(TypeError, self.classifier.prob, document, label)

    def test_prob_all(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        prob_all = self.classifier.prob_all(document.split())
        tests = [
                 ('yes', Fraction(4782969, 6934265)),
                 ('no', Fraction(2151296, 6934265)),
                ]
        for label, prob in tests:
            assert prob_all[label] == prob

    def test_prob_all_not_tokenized(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        assert_raises(TypeError, self.classifier.prob_all, document)

    def test_classify(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'.split()
        label = self.classifier.classify(document)
        assert label == 'yes'

    def test_classify_not_tokenized(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        assert_raises(TypeError, self.classifier.classify, document)
