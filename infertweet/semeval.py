# Copyright (C) 2013 Wesley Baugh
"""Tools for SemEval-2013: Sentiment Analysis in Twitter.

For more information visit <http://www.cs.york.ac.uk/semeval-2013/task2/>.
"""
from collections import namedtuple


def task_b_generator(dataset_file):
    """Parses the Task B dataset and yields LabeledTweet tuples.

    Args:
        dataset_file: File object (or any iterable) containing data in
            the format of the Task B subtask.

    Returns:
        LabeledTweet (a namedtuple) with the following attributes:
            - sid
            - uid
            - label
            - text
    """
    for line in dataset_file:
        LabeledTweet = namedtuple('LabeledTweet', 'sid uid label text')
        line = line.rstrip('\n').split('\t')
        # Fix the `label` field.
        line[2] = line[2].strip('"')
        if line[2] == 'objective-OR-neutral' or line[2] == 'objective':
            line[2] = 'neutral'
        # Only yield if the tweet was downloaded from Twitter.
        if line[3] != 'Not Available':
            yield LabeledTweet(*line)
