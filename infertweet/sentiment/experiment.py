# Copyright (C) 2013 Wesley Baugh
import itertools
from pprint import pprint

from infertweet.semeval import task_b_generator, evaluate
from infertweet.sentiment.constants import (
    TITLES, TEST_SCALE, TWITTER_TEST, SMS_TEST, TWITTER_PREDICT, SMS_PREDICT)


def write_semeval_predictions(experiment, final=False):
    # task2-B-twitter
    with open(TWITTER_TEST) as f, \
            open(TWITTER_PREDICT + ('.final' if final else ''), mode='w') as w:
        for instance in task_b_generator(f):
            sid, uid, label, text = instance
            features = experiment.extractor.extract(instance.text)
            label, probability = experiment._predict(features)
            w.write('\t'.join([sid, uid, label, text]) + '\n')

    # task2-B-SMS
    with open(SMS_TEST) as f, \
            open(SMS_PREDICT + ('.final' if final else ''), mode='w') as w:
        for instance in task_b_generator(f):
            sid, uid, label, text = instance
            features = experiment.extractor.extract(instance.text)
            label, probability = experiment._predict(features)
            w.write('\t'.join([sid, uid, label, text]) + '\n')


def run_experiment(first, second, extractor, chunk_size, first_chunk=0):
    Approach = type('_'.join(x.__name__ for x in first), first, {})
    singular_classifier = Approach(extractor, chunk_size, first_chunk,
                                   test_scale=TEST_SCALE,
                                   evaluator=evaluate)
    print repr(singular_classifier)

    Approach = type('_'.join(x.__name__ for x in second), second, {})
    hierarchical_classifier = Approach(extractor, chunk_size, first_chunk,
                                       test_scale=TEST_SCALE,
                                       evaluator=evaluate)
    print repr(hierarchical_classifier)
    # hierarchical_classifier = OldClassifier(extractor)

    best_performance = None, None, None
    p1, p2 = [], []  # Declare in case the try block raises an exception.
    try:
        for single, hierarchy in itertools.izip(singular_classifier,
                                                hierarchical_classifier):
            new_best = None
            c1, p1 = single
            c2, p2 = hierarchy
            data = dict()
            data[TITLES[0]] = p1
            data[TITLES[0]]['count'] = c1
            # data[TITLES[0]]['vocab'] = singular_classifier.nb._vocab_size#, len(singular_classifier.nb._most_common['positive'].store)
            data[TITLES[0]]['vocab'] = singular_classifier.nb._vocab_size  # , len(singular_classifier.polarity._most_common['positive'].store)
            data[TITLES[1]] = p2
            data[TITLES[1]]['count'] = c2
            data[TITLES[1]]['vocab'] = hierarchical_classifier.polarity._vocab_size  # , len(hierarchical_classifier.polarity._most_common['positive'].store)

            if data[TITLES[0]]['semeval f_measure'] > best_performance[0]:
                new_best = data[TITLES[0]]['semeval f_measure'], singular_classifier, data[TITLES[0]]
                best_performance = new_best
            if data[TITLES[1]]['semeval f_measure'] > best_performance[0]:
                new_best = data[TITLES[1]]['semeval f_measure'], hierarchical_classifier, data[TITLES[1]]
                best_performance = new_best
            # if new_best:
            #     print 'New Best! (see below):', new_best[1].__class__.__name__
            #     pprint(new_best[2])
            #     with open(r"D:\semeval-best.pickle", mode='wb') as f:
            #         f.write(new_best[1].pickle_dumps())
            #     write_semeval_predictions(new_best[1])

            yield data
    except KeyboardInterrupt:
        raise
    finally:
        print 'Final performance:'
        try:
            for label, performance in zip(TITLES, (p1, p2)):
                confusion_matrix = performance['confusionmatrix'].pp()
                # del performance['confusion']
                print label
                pprint(performance)
                print confusion_matrix
        except:
            print 'ERROR: Unavailable.'
