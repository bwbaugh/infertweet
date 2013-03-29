# Copyright (C) 2013 Wesley Baugh
import itertools
import json
from pprint import pprint

from infertweet.config import get_config
from infertweet.corpus.semeval import task_b_generator, evaluate


def write_semeval_predictions(experiment, final=False):
    config = get_config()
    twitter_test = config.get('semeval', 'twitter_test')
    twitter_predict = config.get('semeval', 'twitter_predict')
    sms_test = config.get('semeval', 'sms_test')
    sms_predict = config.get('semeval', 'sms_predict')

    # task2-B-twitter
    with open(twitter_test) as f, \
            open(twitter_predict + ('.final' if final else ''), mode='w') as w:
        for instance in task_b_generator(f):
            sid, uid, label, text = instance
            features = experiment.extractor.extract(instance.text)
            label, probability = experiment._predict(features)
            w.write('\t'.join([sid, uid, label, text]) + '\n')

    # task2-B-SMS
    with open(sms_test) as f, \
            open(sms_predict + ('.final' if final else ''), mode='w') as w:
        for instance in task_b_generator(f):
            sid, uid, label, text = instance
            features = experiment.extractor.extract(instance.text)
            label, probability = experiment._predict(features)
            w.write('\t'.join([sid, uid, label, text]) + '\n')


def run_experiment(first, second, extractor, chunk_size, first_chunk=0):
    config = get_config()
    titles = json.loads(config.get('sentiment', 'titles'))
    test_scale = config.getint('sentiment', 'test_scale')

    Approach = type('_'.join(x.__name__ for x in first), first, {})
    singular_classifier = Approach(extractor, chunk_size, first_chunk,
                                   test_scale=test_scale,
                                   evaluator=evaluate)
    print repr(singular_classifier)

    Approach = type('_'.join(x.__name__ for x in second), second, {})
    hierarchical_classifier = Approach(extractor, chunk_size, first_chunk,
                                       test_scale=test_scale,
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
            data[titles[0]] = p1
            data[titles[0]]['count'] = c1
            # data[titles[0]]['vocab'] = singular_classifier.nb._vocab_size#, len(singular_classifier.nb._most_common['positive'].store)
            data[titles[0]]['vocab'] = singular_classifier.nb._vocab_size  # , len(singular_classifier.polarity._most_common['positive'].store)
            data[titles[1]] = p2
            data[titles[1]]['count'] = c2
            data[titles[1]]['vocab'] = hierarchical_classifier.polarity._vocab_size  # , len(hierarchical_classifier.polarity._most_common['positive'].store)

            if data[titles[0]]['semeval f_measure'] > best_performance[0]:
                new_best = data[titles[0]]['semeval f_measure'], singular_classifier, data[titles[0]]
                best_performance = new_best
            if data[titles[1]]['semeval f_measure'] > best_performance[0]:
                new_best = data[titles[1]]['semeval f_measure'], hierarchical_classifier, data[titles[1]]
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
            for label, performance in zip(titles, (p1, p2)):
                confusion_matrix = performance['confusionmatrix'].pp()
                # del performance['confusion']
                print label
                pprint(performance)
                print confusion_matrix
        except:
            print 'ERROR: Unavailable.'
