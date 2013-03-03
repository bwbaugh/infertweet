# Copyright (C) 2013 Wesley Baugh
"""Tools for text classification."""
import abc
from collections import defaultdict
from fractions import Fraction


class Classifier(object):
    """Abstract base class for classifiers."""
    __metaclass__ = abc.ABCMeta


class MultinomialNB(Classifier):
    """Multinomial Naive Bayes for text classification.

    Attributes:
        laplace: Smoothing parameter >= 0 (default=1).
        labels: Set of all class labels.
        vocabulary: Set of vocabulary across all class labels.
    """
    def __init__(self, *documents):
        """Create a new Multinomial Naive Bayes classifier.
        Args:
            documents: Optional list of document-label pairs for training.
        """
        self.laplace = 1
        # Dictionary of sets of vocabulary by label.
        self._label_vocab = defaultdict(set)
        # Dictionary of times a label has been seen.
        self._label_count = defaultdict(int)
        # Dictionary of number of tokens seen in all documents by label.
        self._label_length = defaultdict(int)
        # Dictionary of times a token has been seen by label.
        self._label_token_count = defaultdict(lambda: defaultdict(int))
        if documents:
            self.train(*documents)

    @property
    def labels(self):
        """Set of all class labels."""
        return set(label for label in self._label_count)

    @property
    def vocabulary(self):
        """Set of vocabulary across all class labels."""
        label_vocab = [self._label_vocab[x] for x in self._label_vocab]
        return set().union(*label_vocab)

    def train(self, *documents):
        """Train the classifier on a document-label pair(s).

        Args:
            documents: Tuple of (document, label) pair(s). Documents
                must be a list of tokens, The label is a string for the
                class label.
        """
        for document, label in documents:
            # Python 3: isinstance(document, str)
            if isinstance(document, basestring):
                raise TypeError('Documents must be a list of tokens')
            self._label_count[label] += 1
            for token in document:
                self._label_vocab[label].add(token)
                self._label_token_count[label][token] += 1
                self._label_length[label] += 1

    def prior(self, label):
        """Prior probability of a label."""
        if label not in self.labels:
            raise KeyError(label)
        total = sum(self._label_count[x] for x in self._label_count)
        return Fraction(self._label_count[label], total)

    def conditional(self, token, label):
        """Conditional probability for a token given a label."""
        # Note we use [Laplace smoothing][laplace].
        # [laplace]: https://en.wikipedia.org/wiki/Additive_smoothing
        if label not in self.labels:
            raise KeyError(label)

        # Times token seen across all documents in a label.
        numer = self.laplace
        # Avoid creating an entry if the term has never been seen
        if token in self._label_token_count[label]:
            numer += self._label_token_count[label][token]
        denom = self._label_length[label] + (len(self.vocabulary) *
                                             self.laplace)
        return Fraction(numer, denom)

    def score(self, document, label):
        """Multinomial score of a document given a label."""
        if isinstance(document, basestring):
            raise TypeError('Documents must be a list of tokens')
        score = self.prior(label)
        for token in document:
            score *= self.conditional(token, label)
        return score

    def _compute_scores(self, document):
        """Compute the multinomial score of a document for all labels."""
        return {x: self.score(document, x) for x in self.labels}

    def prob_all(self, document):
        """Probability of a document for all labels."""
        score = self._compute_scores(document)
        total = sum(score[x] for x in score)
        return {label: Fraction(score[label], total) for label in self.labels}

    def prob(self, document, label):
        """Probability of a document given a label."""
        prob = self.prob_all(document)[label]
        return prob

    def classify(self, document):
        """Class label with maximum probability for a document."""
        prob = self.prob_all(document)
        return max(prob, key=prob.get)
