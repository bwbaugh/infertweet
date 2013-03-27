# Copyright (C) 2013 Wesley Baugh
"""Art Festival tweets with sentiment labels."""
from infer.experiment import Experiment

from infertweet.config import get_config


class TestArtFestival(Experiment):
    def _test_data(self):
        for instance in parse_artfestival():
            yield instance


class TrainArtFestival(Experiment):
    def _train_data(self):
        for instance in parse_artfestival():
            yield instance


def parse_artfestival():
    config = get_config()
    with open(config.get('art_festival', 'corpus')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            data = line.rstrip().split('\t')
            author, tweet, sentiment = [x.strip() for x in data[:3]]
            yield Experiment.DataInstance(tweet, sentiment)
