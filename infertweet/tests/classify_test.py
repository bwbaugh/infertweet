# Copyright (C) 2013 Wesley Baugh
from __future__ import division

from copy import deepcopy
from fractions import Fraction
from nose.tools import assert_raises, assert_almost_equal

from infertweet.classify import MultinomialNB, evaluate


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


class TestEvaluate(object):
    def setup(self):
        self.reference = 'DET NN VB DET JJ NN NN IN DET NN'.split()
        self.test = 'DET VB VB DET NN NN NN IN DET NN'.split()
        self.performance = evaluate(self.reference, self.test)
        self.labels = set(self.reference + self.test)
        self.expected_recall = [('DET', 1),
                                ('IN', 1),
                                ('JJ', 0),
                                ('NN', 0.75),
                                ('VB', 1)]
        self.expected_precision = [('DET', 1),
                                   ('IN', 1),
                                   ('JJ', 0),
                                   ('NN', 0.75),
                                   ('VB', 0.5)]
        self.expected_f_measure = [('DET', 1),
                                   ('IN', 1),
                                   ('JJ', 0),
                                   ('NN', 0.75),
                                   ('VB', 2 / 3)]

    def test_evaluate_class_metrics(self):
        expected = []
        for label in self.labels:
            expected.append('f-{0}'.format(label))
            expected.append('recall-{0}'.format(label))
            expected.append('precision-{0}'.format(label))
        failed = []
        for metric in expected:
            if metric not in self.performance.keys():
                failed.append(metric)
        assert not failed

    def test_confusion_matrix(self):
        confusion = self.performance['confusionmatrix']
        result = confusion.pp()
        expected = ('    | D         |\n    | E I J N V |\n    | T N J '
                    'N B |\n----+-----------+\nDET |<3>. . . . |\n IN |'
                    ' .<1>. . . |\n JJ | . .<.>1 . |\n NN | . . .<3>1 |'
                    '\n VB | . . . .<1>|\n----+-----------+\n(row = ref'
                    'erence; col = test)\n')
        assert result == expected

    def test_accuracy(self):
        result = self.performance['accuracy']
        assert result == 0.8

    def test_recall_class(self):
        failed = []
        for label, result in self.expected_recall:
            computed = self.performance['recall-{0}'.format(label)]
            if computed != result:
                failed.append((label, result, computed))
        assert not failed

    def test_recall_average(self):
        result = self.performance['average recall']
        assert_almost_equal(result, 0.75)

    def test_recall_weighted(self):
        expected = [(label, value * self.reference.count(label)) for
                    label, value in self.expected_recall]
        expected = sum(value for label, value in expected) / len(self.reference)
        result = self.performance['weighted recall']
        assert_almost_equal(result, expected)

    def test_precision_class(self):
        failed = []
        for label, result in self.expected_precision:
            computed = self.performance['precision-{0}'.format(label)]
            if computed != result:
                failed.append((label, result, computed))
        assert not failed

    def test_precision_average(self):
        result = self.performance['average precision']
        assert_almost_equal(result, 0.65)

    def test_precision_weighted(self):
        expected = [(label, value * self.reference.count(label)) for
                    label, value in self.expected_precision]
        expected = sum(value for label, value in expected) / len(self.reference)
        result = self.performance['weighted precision']
        assert_almost_equal(result, expected)

    def test_f_measure_class(self):
        failed = []
        for label, result in self.expected_f_measure:
            computed = self.performance['f-{0}'.format(label)]
            try:
                assert_almost_equal(computed, result)
            except AssertionError:
                failed.append((label, result, computed))
        assert not failed

    def test_f_measure_beta(self):
        failed = []
        for beta in (0.5, 2):
            for x, y in zip(self.expected_precision, self.expected_recall):
                label, precision, recall = x[0], x[1], y[1]
                performance = evaluate(self.reference, self.test, beta=beta)
                result = performance['f-{0}'.format(label)]
                if precision == 0 and recall == 0:
                    expected = 0
                else:
                    expected = (((1 + beta ** 2) * precision * recall) /
                                (((beta ** 2) * precision) + recall))
                try:
                    assert_almost_equal(result, expected)
                except AssertionError:
                    failed.append((label, beta, result, expected))
        assert not failed

    def test_f_measure_average(self):
        result = self.performance['average f_measure']
        assert_almost_equal(result, (2.75 + (2 / 3)) / 5)

    def test_f_measure_weighted(self):
        expected = [('DET', 1),
                    ('IN', 1),
                    ('JJ', 0),
                    ('NN', 0.75),
                    ('VB', 2 / 3)]
        expected = [(label, value * self.reference.count(label)) for
                    label, value in expected]
        expected = sum(value for label, value in expected) / len(self.reference)
        result = self.performance['weighted f_measure']
        assert_almost_equal(result, expected)
