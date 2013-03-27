# Copyright (C) 2013 Wesley Baugh
"""Stanford sentiment corpus.

See: <http://help.sentiment140.com/for-students>
"""
from infer.experiment import Experiment

from infertweet.config import get_config


class TrainStanford(Experiment):
    def _train_data(self):
        config = get_config()
        with open(config.get('stanford', 'corpus')) as f:
            for line in f:
                line = line.split(',', 5)
                if len(line) != 6:
                    continue
                sentiment = int(line[0][1:-1])
                if sentiment == 0:
                    sentiment = 'negative'
                elif sentiment == 2:
                    sentiment = 'neutral'
                elif sentiment == 4:
                    sentiment = 'positive'
                text = line[5]
                # Strip surrounding '"'
                text = text[1:-1]
                yield self.DataInstance(text, sentiment)
