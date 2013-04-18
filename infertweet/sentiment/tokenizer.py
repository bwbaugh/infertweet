# Copyright (C) 2013 Wesley Baugh
"""Tokenizer and related helper functions."""
import re


def pre_usernames(document):
    """Fold @usernames to `__USERNAME__`."""
    return re.sub(r'@([A-Za-z0-9_]+)', '__USERNAME__', document)


def pre_urls(document):
    """Fold URLs to `__URL__`."""
    document = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.]"""
        r"""[a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+"""
        r"""\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"""
        r"""[^\s`!()\[\]{};:'".,<>?]))""", '__URL__', document)
    return document


def pre_dates(document):
    """Fold detected dates to `__DATE__`."""
    # Source: http://stackoverflow.com/a/3810334/1988505
    # capital R because of sublime-text-2 syntax highlighting problem.
    document = re.sub(
        R"""(?ix)             # case-insensitive, verbose regex
        \b                    # match a word boundary
        (?:                   # match the following three times:
         (?:                  # either
          \d+                 # a number,
          (?:\.|st|nd|rd|th)* # followed by a dot, st, nd, rd, or th (optional)
          |                   # or a month name
          (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)
         )
         [\s./-]*             # followed by a date separator or whitespace (opt)
        ){3}                  # do this three times
        \b                    # and end at a word boundary.""",
        ' __DATE__ ', document)
    return document


def shorten_repeated_chars(token):
    """Shorten repeated chars (haaaaaaate -> haate)."""
    return re.sub(r"(\w)\1{2,}", r'\1\1', token)


def preprocess(document):
    """Preprocess a document."""
    pre_urls(document)
    pre_dates(document)


def analyze_tokens(tokens):
    """Analyze tokens to extract additional features."""
    features = []
    return features


def process_tokens(tokens):
    """Process and filter tokens."""
    for i, token in enumerate(tokens):
        token = shorten_repeated_chars(token)
    tokens = [x for x in tokens if x]
    return tokens


def tokenizer(document):
    """Create feature tokens from an unprocessed document.

    Args:
        document: Original raw string of a document.

    Returns:
        Tuple containing ordered tokens suitable for n-grams, and
        unordered features that were extracted and are not suitable for
        use with n-grams.
    """
    preprocess(document)
    tokens = document.split(' ')
    analysis_features = analyze_tokens(tokens)
    tokens = process_tokens(tokens)
    return tokens, analysis_features
