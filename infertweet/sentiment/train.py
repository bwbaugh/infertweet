# Copyright (C) 2013 Wesley Baugh
"""Sentiment analysis for SemEval-2013."""
# Normally this file would be named `__main__.py`, however there is a
# bug with multiprocessing that prevents it from working under Windows.
import json
import multiprocessing
from collections import namedtuple

from infer.nlp import FeatureExtractor

import infertweet.corpus.semeval as semeval
from infertweet.config import get_config
from infertweet.sentiment.experiment import (
    SingleClassifier, HierarchicalClassifier, run_experiment, tokenizer,
    parse_performance)
from infertweet.sentiment.plot import start_plot


Pickled = namedtuple('Pickled', 'extractor classifier')


def main():
    plot_queue = multiprocessing.Queue()
    confusion_queue = multiprocessing.Queue()
    start_plot(plot_queue, confusion_queue)

    first = (SingleClassifier, semeval.TrainSemEvalSelfLearning,
             semeval.TestSemEval)
    second = (HierarchicalClassifier, semeval.TrainSemEvalSelfLearning,
              semeval.TestSemEval)

    extractor = FeatureExtractor(tokenizer=tokenizer)
    extractor.min_n, extractor.max_n = 1, 2

    config = get_config()
    chunk_size = config.getint('sentiment', 'chunk_size')
    first_chunk = config.getint('sentiment', 'first_chunk')
    titles = json.loads(config.get('sentiment', 'titles'))

    experiment = run_experiment(first, second, extractor, chunk_size,
                                first_chunk)

    try:
        for data in experiment:
            data[titles[0]] = parse_performance(data[titles[0]])
            data[titles[1]] = parse_performance(data[titles[1]])
            plot_queue.put(data)
            confusion_queue.put(data)
            print data[titles[0]]['count'], data[titles[0]]['SemEval'], data[titles[1]]['SemEval'], data[titles[0]]['vocab'], data[titles[1]]['vocab']
    except KeyboardInterrupt:
        pass
    finally:
        plot_queue.put(None)
        plot_queue.close()
        confusion_queue.put(None)
        confusion_queue.close()

    print 'Done processing.'


if __name__ == '__main__':
    main()
