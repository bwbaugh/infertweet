# Copyright (C) 2013 Brian Wesley Baugh
"""Natural language processing (NLP) utility functions."""


class FeatureExtractor(object):
    """Extract features from text for use in classification.

    Attributes:
        min_n: Minimum sized n-gram to extract.
        max_n: Maximum sized n-gram to extract.
    """
    def __init__(self, min_n=None, max_n=None):
        """Create a new FeatureExtractor object.

        Args:
            min_n: Minimum sized n-gram to extract > 0. (default = None)
            max_n: Maximum sized n-gram to extract > 0. (default = None)
        """
        self.min_n, self.max_n = min_n, max_n

    def extract(self, document):
        """Extract features from a document.

        Args:
            document: List of tokens, which can be any hashable object
                but are usually strings.

        Returns:
            A list of features extracted from the document, using the
            attributes of the object to determine what types of features
            to extract. A feature may appear more than once in the list.
        """
        features = []
        if self.min_n is not None and self.max_n is not None:
            features.extend(self.extract_ngrams(document))
        return features

    def extract_ngrams(self, document):
        """Extracts n-gram tuples from a document.

        Args:
            document: List of tokens.
            self.min_n: Minimum sized n-gram to extract.
            self.max_n: Maximum sized n-gram to extract.

        Returns:
            List of n-gram tuples. If n-grams of size greater than 2 are
            extracted, then a special token will be put at the beginning
            of the document '__start__', and a special token at the end
            of the document '__end__'.
            Example:
                Unigrams:
                    document = 'I am so happy about this project'.split()
                    [('I'), ('am'), ('so'), ('happy'), ('about'),
                     ('this'), ('project')]
                Bigrams:
                    document = 'this is a test'.split()
                    [('__start__', 'this'), ('this', 'is'), ('is', 'a'),
                     ('a', 'test'), ('test', '__end__')]
        """
        if self.min_n <= 0 or self.max_n <= 0 or self.min_n > self.max_n:
            message = '0 < min_n ({0}) <= max_n {1} not satisfied'
            raise ValueError(message.format(self.min_n, self.max_n))

        def wrap_start_end_tokens(document):
            yield '__start__'
            for token in document:
                yield token
            yield '__end__'

        document = list(wrap_start_end_tokens(document))

        # Inspiration:
        # http://stackoverflow.com/questions/7591258/fast-n-gram-calculation
        ngrams = []
        n_tokens = len(document)
        for i in xrange(n_tokens):
            for j in xrange(i + self.min_n, min(n_tokens, i + self.max_n) + 1):
                # ngram = ' '.join(document[i:j])
                ngram = tuple(document[i:j])
                if len(ngram) > 1 or (ngram[0] != '__start__' and
                                      ngram[0] != '__end__'):
                    ngrams.append(ngram)

        return ngrams
