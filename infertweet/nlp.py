# Copyright (C) 2013 Brian Wesley Baugh
"""Natural language processing (NLP) utility functions."""


class FeatureExtractor(object):
    """Extract features from text for use in classification.

    Attributes:
        min_n: Minimum sized n-gram to extract.
        max_n: Maximum sized n-gram to extract.
        tokenize: Function that tokenizes a document.
    """
    def __init__(self, min_n=None, max_n=None, tokenizer=None):
        """Create a new FeatureExtractor object.

        Args:
            min_n: Minimum sized n-gram to extract > 0. (default = None)
            max_n: Maximum sized n-gram to extract > 0. (default = None)
            tokenizer: Function that tokenizes a document.
        """
        self.min_n, self.max_n = min_n, max_n
        if tokenizer:
            self.tokenize = tokenizer
        else:
            self.tokenize = str.split

    def extract(self, document):
        """Extract features from a document.

        Args:
            document: An untokenized document string.

        Returns:
            A list of features extracted from the document, using the
            attributes of the object to determine what types of features
            to extract. A feature may appear more than once in the list.
            Note: The document is first tokenized using `self.tokenize`.
        """
        tokens = self.tokenize(document)
        features = []
        if self.min_n is not None and self.max_n is not None:
            features.extend(self.extract_ngrams(tokens))
        return features

    def extract_ngrams(self, tokens):
        """Extracts n-gram tuples from a list of tokens.

        Args:
            tokens: List of tokens.
            self.min_n: Minimum sized n-gram to extract.
            self.max_n: Maximum sized n-gram to extract.

        Returns:
            List of n-gram tuples. If n-grams of size greater than 2 are
            extracted, then a special token will be put at the beginning
            of the token-list '__start__', and a special token at the
            end of the token-list '__end__'.
            Example:
                Unigrams:
                    tokens = 'I am so happy about this project'.split()
                    [('I'), ('am'), ('so'), ('happy'), ('about'),
                     ('this'), ('project')]
                Bigrams:
                    tokens = 'this is a test'.split()
                    [('__start__', 'this'), ('this', 'is'), ('is', 'a'),
                     ('a', 'test'), ('test', '__end__')]
        """
        if self.min_n <= 0 or self.max_n <= 0 or self.min_n > self.max_n:
            message = '0 < min_n ({0}) <= max_n {1} not satisfied'
            raise ValueError(message.format(self.min_n, self.max_n))

        def wrap_start_end_tokens(tokens):
            yield '__start__'
            for token in tokens:
                yield token
            yield '__end__'

        tokens = list(wrap_start_end_tokens(tokens))

        # Inspiration:
        # http://stackoverflow.com/questions/7591258/fast-n-gram-calculation
        ngrams = []
        n_tokens = len(tokens)
        for i in xrange(n_tokens):
            for j in xrange(i + self.min_n, min(n_tokens, i + self.max_n) + 1):
                # ngram = ' '.join(tokens[i:j])
                ngram = tuple(tokens[i:j])
                if len(ngram) > 1 or (ngram[0] != '__start__' and
                                      ngram[0] != '__end__'):
                    ngrams.append(ngram)

        return ngrams
