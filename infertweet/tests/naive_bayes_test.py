# Copyright (C) 2013 Wesley Baugh
from copy import deepcopy
from fractions import Fraction
from nose.tools import assert_raises

from infertweet.naive_bayes import MultinomialNB


class TestMultinomialNB(object):
    # This test uses the examples provided by:
    # http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
    def setup(self):
        self.training_docs = [('Chinese Bejing Chinese', 'yes'),
                              ('Chinese Chinese Shanghai', 'yes'),
                              ('Chinese Macao', 'yes'),
                              ('Tokyo Japan Chinese', 'no')]
        self.training_docs = [(x.split(), y) for x, y in self.training_docs]
        self.classifier = MultinomialNB(*self.training_docs)
        self.make_snapshot()

    def make_snapshot(self):
        self.orig_label_count = deepcopy(self.classifier._label_count)
        self.orig_label_vocab = deepcopy(self.classifier._label_vocab)
        self.orig_label_token_count = deepcopy(self.classifier
                                               ._label_token_count)
        self.orig_label_length = deepcopy(self.classifier._label_length)

    def assert_snapshot_identical(self):
        """Call if classifier's internals shouldn't have changed."""
        assert self.orig_label_count == self.classifier._label_count
        assert self.orig_label_vocab == self.classifier._label_vocab
        assert self.orig_label_token_count == self.classifier._label_token_count
        assert self.orig_label_length == self.classifier._label_length

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
        document = ('one document not tokenized', 'label')
        assert_raises(TypeError, self.classifier.train, document)
        self.assert_snapshot_identical()

    def test_labels(self):
        expected = set(['yes', 'no'])
        assert self.classifier.labels == expected
        self.assert_snapshot_identical()

    def test_vocabulary(self):
        expected = set(['Chinese', 'Bejing', 'Shanghai', 'Macao', 'Tokyo',
                        'Japan'])
        assert self.classifier.vocabulary == expected
        self.assert_snapshot_identical()

    def test_label_token_count(self):
        tests = [('yes', 'Chinese', 5),
                 ('no', 'Chinese', 1),
                 ('no', 'Japan', 1)]
        for label, token, count in tests:
            assert self.classifier._label_token_count[label][token] == count
        assert 'Japan' not in self.classifier._label_token_count['yes']
        self.assert_snapshot_identical()

    def test_prior(self):
        tests = [('yes', Fraction(3, 4)),
                 ('no', Fraction(1, 4))]
        for label, prob in tests:
            assert self.classifier.prior(label) == prob
        self.assert_snapshot_identical()

    def test_prior_unseen_label(self):
        assert_raises(KeyError, self.classifier.prior, '__unseen__')
        self.assert_snapshot_identical()

    def test_conditional(self):
        tests = [('Chinese', 'yes', Fraction(6, 14)),
                 ('Japan', 'yes', Fraction(1, 14)),
                 ('Chinese', 'no', Fraction(2, 9)),
                 ('Tokyo', 'no', Fraction(2, 9)),
                 ('Japan', 'no', Fraction(2, 9)),
                 ('__invalid__', 'yes', Fraction(1, 14)),
                 ('__invalid__', 'no', Fraction(1, 9))]
        for token, label, prob in tests:
            result = self.classifier.conditional(token, label)
            assert result == prob
        self.assert_snapshot_identical()

    def test_conditional_laplace(self):
        self.classifier.laplace = 2
        tests = [('Chinese', 'yes', Fraction(7, 20)),
                 ('Japan', 'yes', Fraction(1, 10)),
                 ('Chinese', 'no', Fraction(1, 5)),
                 ('Tokyo', 'no', Fraction(1, 5)),
                 ('Japan', 'no', Fraction(1, 5)),
                 ('__invalid__', 'yes', Fraction(1, 10)),
                 ('__invalid__', 'no', Fraction(2, 15))]
        for token, label, prob in tests:
            result = self.classifier.conditional(token, label)
            assert result == prob
        self.assert_snapshot_identical()

    def test_conditional_unseen_token(self):
        self.classifier.conditional('__unseen__', 'yes')
        assert '__unseen__' not in self.classifier._label_token_count['yes']
        self.assert_snapshot_identical()

    def test_conditional_unseen_label(self):
        assert_raises(KeyError, self.classifier.conditional, '__unseen__',
                      '__unseen__')
        assert '__unseen__' not in self.classifier._label_token_count
        self.assert_snapshot_identical()

    def test_score(self):
        tests = [('Chinese Chinese Chinese Tokyo Japan', 'yes',
                  Fraction(3, 4) * Fraction(3, 7) * Fraction(3, 7) *
                  Fraction(3, 7) * Fraction(1, 14) * Fraction(1, 14)),
                 ('Chinese Chinese Chinese Tokyo Japan', 'no',
                  Fraction(1, 4) * Fraction(2, 9) * Fraction(2, 9) *
                  Fraction(2, 9) * Fraction(2, 9) * Fraction(2, 9))]
        for document, label, score in tests:
            result = self.classifier.score(document.split(), label)
            assert result == score
        self.assert_snapshot_identical()

    def test_score_not_tokenized(self):
        document, label = 'Chinese Chinese Chinese Tokyo Japan', 'yes'
        assert_raises(TypeError, self.classifier.score, document, label)
        self.assert_snapshot_identical()

    def test_prob(self):
        tests = [('Chinese Chinese Chinese Tokyo Japan', 'yes',
                  Fraction(4782969, 6934265)),
                 ('Chinese Chinese Chinese Tokyo Japan', 'no',
                  Fraction(2151296, 6934265))]
        for document, label, prob in tests:
            result = self.classifier.prob(document.split(), label)
            assert result == prob
        self.assert_snapshot_identical()

    def test_prob_not_tokenized(self):
        document, label = 'Chinese Chinese Chinese Tokyo Japan', 'yes'
        assert_raises(TypeError, self.classifier.prob, document, label)
        self.assert_snapshot_identical()

    def test_prob_all(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        prob_all = self.classifier.prob_all(document.split())
        tests = [('yes', Fraction(4782969, 6934265)),
                 ('no', Fraction(2151296, 6934265))]
        for label, prob in tests:
            assert prob_all[label] == prob
        self.assert_snapshot_identical()

    def test_prob_all_not_tokenized(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        assert_raises(TypeError, self.classifier.prob_all, document)
        self.assert_snapshot_identical()

    def test_classify(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'.split()
        label = self.classifier.classify(document)
        assert label == 'yes'
        self.assert_snapshot_identical()

    def test_classify_not_tokenized(self):
        document = 'Chinese Chinese Chinese Tokyo Japan'
        assert_raises(TypeError, self.classifier.classify, document)
        self.assert_snapshot_identical()
