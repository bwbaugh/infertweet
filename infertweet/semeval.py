# Copyright (C) 2013 Wesley Baugh
"""Tools for SemEval-2013: Sentiment Analysis in Twitter.

For more information visit <http://www.cs.york.ac.uk/semeval-2013/task2/>.
"""
import ast
from collections import namedtuple

from unidecode import unidecode

from infertweet import classify


def task_b_generator(dataset_file):
    """Parses the Task B dataset and yields LabeledTweet tuples.

    Args:
        dataset_file: File object (or any iterable) containing data in
            the format of the Task B subtask.

    Returns:
        LabeledTweet (a namedtuple) with the following attributes:
            - sid: String.
            - uid: String.
            - label: String in ['positive', 'negative', 'neutral'] or None.
            - text: String.
    """
    for line in dataset_file:
        LabeledTweet = namedtuple('LabeledTweet', 'sid uid label text')
        line = line.rstrip('\n').split('\t')
        sid, uid, label, text = line
        label = label.strip('"')
        if label == 'objective-OR-neutral' or label == 'objective':
            label = 'neutral'
        if label == 'unknwn':
            label = None
        # Text might be in the python style `repr()` encoding.
        try:
            text = ast.literal_eval(''.join(['u"', text, '"']))
        except SyntaxError:
            pass
        text = unidecode(text)
        if text != 'Not Available':
            yield LabeledTweet(sid, uid, label, text)


def evaluate(reference, test, beta=1):
    """Compute score for SemEval and various performance metrics.

    Args:
        reference: An ordered list of correct class labels.
        test: A corresponding ordered list of class labels to evaluate.
        beta: A float parameter for F-measure (default = 1).

    Returns:
        A dictionary with an entry for each metric. An additional entry
        is made with the key 'semeval f_measure', which is the
        performance metric used by SemEval-2013.
    """
    performance = classify.evaluate(reference, test, beta)
    semeval = (performance['f-positive'] + performance['f-negative']) / 2
    performance['semeval f_measure'] = semeval
    return performance
