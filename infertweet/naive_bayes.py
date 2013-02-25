# Copyright (C) 2013 Wesley Baugh
"""Naive Bayes for text classification."""
from collections import defaultdict
from fractions import Fraction


class MultinomialNB(object):
    """Multinomial Naive Bayes for text classification.

    Attributes:
        label_vocab: Dictionary of sets of vocabulary by label.
        label_count: Dictionary of times a label has been seen.
        label_length: Dictionary of number of tokens seen in all
            documents by label.
        label_token_count: Dictionary of times a token has been seen by
            label.
    """
    def __init__(self, *documents):
        """Create a new Multinomial Naive Bayes classifier.
        Args:
            documents: Optional list of document-label pairs for training.
        """
        self.laplace = 1
        self.label_vocab = defaultdict(set)
        self.label_count = defaultdict(int)
        self.label_length = defaultdict(int)
        self.label_token_count = defaultdict(lambda: defaultdict(int))
        if documents:
            self.train(*documents)

    @property
    def labels(self):
        """Set of all class labels."""
        return set(label for label in self.label_count)

    @property
    def vocabulary(self):
        """Vocabulary across all class labels."""
        label_vocab = [self.label_vocab[x] for x in self.label_vocab]
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
            self.label_count[label] += 1
            for token in document:
                self.label_vocab[label].add(token)
                self.label_token_count[label][token] += 1
                self.label_length[label] += 1

    def prior(self, label):
        """Prior probability of a label."""
        total = sum(self.label_count[x] for x in self.label_count)
        return Fraction(self.label_count[label], total)

    def conditional(self, token, label):
        """Conditional probability for a token given a label."""
        # Note we use [Laplace smoothing][laplace].
        # [laplace]: https://en.wikipedia.org/wiki/Additive_smoothing

        # Times token seen across all documents in a label.
        numer = self.label_token_count[label][token] + self.laplace
        denom = self.label_length[label] + (len(self.vocabulary) * self.laplace)
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
