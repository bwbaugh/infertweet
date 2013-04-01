# Copyright (C) 2013 Wesley Baugh
import itertools
import json
import re
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import namedtuple
from pprint import pprint

from infer.classify import MultinomialNB
from infer.experiment import Experiment

import infertweet.corpus.semeval as semeval
import infertweet.corpus.stanford as stanford140
from infertweet.config import get_config
from infertweet.corpus.semeval import task_b_generator, evaluate


Pickled = namedtuple('Pickled', 'extractor classifier')


class TrainWikipedia(Experiment):
    def _train_data(self):
        with open(r"R:\_Other\Wikipedia\simplewiki-20121209-pages-articles.xml.txt") as f:
            for article in f:
                text = article.split('\t', 3)[2].decode('utf-8')
                for paragraph in text.split(u'\u2029'):
                    for sentence in paragraph.split(u'\u2028'):
                        sentence = sentence.strip()
                        if sentence:
                            yield self.DataInstance(sentence, 'neutral')


class TrainSemEvalWithStanford(semeval.TrainSemEval, stanford140.TrainStanford):
    def _train_data(self):
        for x in semeval.TrainSemEval._train_data(self):
            yield x
        for x in stanford140.TrainStanford._train_data(self):
            yield x


class TrainZipStanfordWikipedia(semeval.TrainSemEval,
                                stanford140.TrainStanford, TrainWikipedia):
    def _train_data(self):
        for x in semeval.TrainSemEval._train_data(self):
            yield x
        stanford = stanford140.TrainStanford._train_data(self)
        wikipedia = TrainWikipedia._train_data(self)
        while 1:
            yield next(stanford)
            for _ in xrange(64):
                yield next(wikipedia)


class SingleClassifier(Experiment):
    """Train a single classifier and return performance on a test set."""

    def _setup(self):
        self.nb = MultinomialNB()
        # self.nb.top = 100000

    def _predict(self, features):
        return self.nb.classify(features)

    def _train(self, features, label):
        self.nb.train((features, label))

    def pickle_dumps(self):
        pickled = Pickled(self.extractor, self.nb)
        return pickle.dumps(pickled, pickle.HIGHEST_PROTOCOL)


class SingleClassifierUniformPrior(SingleClassifier):
    def _setup(self):
        super(SingleClassifier, self)._setup()
        self.nb.prior = lambda x: 1.0 / len(self.nb.labels)  # Will this work?


class SingleClassifierNeutralCertainty(SingleClassifier):
    def _predict(self, features):
        prob = self.nb.prob_all(features)
        if prob['neutral'] > 0.2:
            return 'neutral', prob['neutral']
        if prob['positive'] > prob['negative']:
            return 'positive', prob['positive']
        else:
            return 'negative', prob['negative']


class SingleClassifierByCount(SingleClassifier):
    def _predict(self, features):
        foo = {'positive': 0, 'negative': 0, 'neutral': 0}
        for feature in features:
            label, probability = self.nb.classify(feature)
            if probability > 0.8:
                foo[label] += 1
        return max(foo, key=foo.get), 1


class HierarchicalClassifier(Experiment):
    """Train a single classifier and return performance on a test set."""

    def _setup(self):
        self.subjective, self.polarity = MultinomialNB(), MultinomialNB()
        # self.subjective.top, self.polarity.top = 100000, 100000
        # subjective.prior = lambda x: 0.5
        # polarity.prior = lambda x: 0.5

    def _predict(self, features):
        label, probability = self.subjective.classify(features)
        if label == 'neutral':
            return label, probability
        else:
            return self.polarity.classify(features)

    def _train(self, features, label):
        if label != 'neutral':
            assert label in set(['positive', 'negative'])
            self.polarity.train((features, label))
            label = 'subjective'
        assert label in set(['neutral', 'subjective'])
        if sum(self.subjective._label_count[x] for x in self.subjective._label_count) < 8751:
            self.subjective.train((features, label))

    def pickle_dumps(self):
        pickled = Pickled(self.extractor, (self.subjective, self.polarity))
        return pickle.dumps(pickled, pickle.HIGHEST_PROTOCOL)


class HierarchicalClassifierByCount(HierarchicalClassifier):
    def _predict(self, features):
        foo = {'positive': 0, 'negative': 0, 'neutral': 0}
        for feature in features:
            label, probability = self.subjective.classify(feature)
            if label == 'neutral':
                if probability > 0.8:
                    foo[label] += 1
            else:
                label, probability = self.polarity.classify(feature)
                if probability > 0.8:
                    foo[label] += 1
        return max(foo, key=foo.get), 1


class OldClassifier(Experiment):
    def _setup(self):
        import old_classify

        def extract(x):
            old_classify.filter_text(x)
            old_classify.regularlize_text(x)
            features = old_classify.extract_features(x)
            return features
        self.extractor.extract = extract
        # tokens = old_classify.tokenizer.tokenize(x)
        # tokens = old_classify.regularlize_tokens(tokens)

        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # twitter-sentiment_classifier.1650000.pickle
        # with open(r"R:\_Other\Twitter\TwitterCorpus\results_sentiment\unbalanced_1-gram_stopword\twitter-sentiment_classifier.5000.pickle", mode='rb') as f:
        with open(r"R:\_Other\Twitter\TwitterCorpus\results_sentiment\unbalanced_1-gram_stopword\twitter-sentiment_classifier.1650000.pickle", mode='rb') as f:
            self.classifier = pickle.load(f)
        self.subjective = MultinomialNB()

    def _predict(self, features):
        label, probability = self.subjective.classify(features)
        if label == 'neutral':
            return label, probability
        sentiment = self.classifier.prob_classify(features)
        prob, neg, = sentiment.prob('pos'), sentiment.prob('neg')
        if prob > neg:
            return 'positive', prob
        else:
            return 'negative', neg

    def _train_func(self, features, label):
        if label != 'neutral':
            label = 'subjective'
        assert label in set(['neutral', 'subjective'])
        self.subjective.train((features, label))


def parse_performance(performance):
        """Worker process that generates classifier performance data."""
        data = dict()
        data['SemEval'] = performance['semeval f_measure']
        data['Positive'] = performance['f-positive']
        data['Negative'] = performance['f-negative']
        data['Neutral'] = performance['f-neutral']
        data['Accuracy'] = performance['accuracy']
        data['confusion'] = performance['confusionmatrix']
        data['count'] = performance['count']
        data['vocab'] = performance['vocab']
        return data


def tokenizer(document):
    # USERNAMES ( !!! VERY BAD DO NOT USE !!! )
    # document = re.sub(r'@([A-Za-z0-9_]+)', '__USERNAME__', document)
    # URL GOOD
    document = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.]"""
        r"""[a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+"""
        r"""\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"""
        r"""[^\s`!()\[\]{};:'".,<>?]))""", '__URL__', document)
    # DATE
    # "big" boost at 150k (0.5723 semeval, space around, 0.570 w/o)
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

    # exclamation_count = document.count('!')
    tokens = document.split(' ')
    # NEGATE = "not n't".split(' ')
    # previous_was_not = False
    # REPEATED CHARS CURRENTLY REDUCES PERFORMANCE
    # repeated_chars = []
    for i, token in enumerate(tokens):
        # Shorten repeated chars (haaaaaaate -> haate)
        # SUB SO FAR BEST at 135k
        token, num_subs = re.subn(r"(\w)\1{2,}", r'\1\1', token)
        # if num_subs:
        #     repeated_chars.append('__num_subs({0})__'.format(num_subs))
        # if previous_was_not and token not in NEGATE:
        #     token = 'not_' + token
        # previous_was_not = token in NEGATE
        # FLOAT mixed results (0.573 semev at 135k)
        # try:
        #     float(token)
        # except:
        #     pass
        # else:
        #     token = '__FLOAT__'
        # tokens[i] = token
    tokens = [x for x in tokens if x]
    # tokens.extend(repeated_chars)
    # EXCLAMATION (hurt performance, down to 0.527 semev)
    # tokens.append('__EXCLAMATION({0})__'.format(exclamation_count))
    return tokens


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
