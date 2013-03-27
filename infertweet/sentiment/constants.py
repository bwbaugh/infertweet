# Copyright (C) 2013 Wesley Baugh
"""Constants used throughout the sentiment package."""

# TODO(bwbaugh): Convert this entire module into a configuration file.

FIRST_CHUNK = 8100
CHUNK_SIZE = 5000
TEST_SCALE = 1  # divided by this.

TWITTER_TEST = r"C:\Users\AE5NS.AE5NS-D\Dropbox\School\Courses - Current\CSCE 6933 - Semi-Supervised Learning\SemEval-2013\Test\submission\twitter-test-input-B.tsv"
SMS_TEST = r"C:\Users\AE5NS.AE5NS-D\Dropbox\School\Courses - Current\CSCE 6933 - Semi-Supervised Learning\SemEval-2013\Test\submission\sms-test-input-B.tsv"

TWITTER_PREDICT = r"D:\task2-bwbaugh-B-twitter-unconstrained.output"
SMS_PREDICT = r"D:\task2-bwbaugh-B-SMS-unconstrained.output"

TITLES = ('Single Classifier', 'Hierarchical Classifier')
LABELS = ('SemEval', 'Positive', 'Negative', 'Neutral', 'Accuracy')
