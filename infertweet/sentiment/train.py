# Copyright (C) 2013 Wesley Baugh
"""Sentiment analysis for SemEval-2013."""
# Normally this file would be named `__main__.py`, however there is a
# bug with multiprocessing that prevents it from working under Windows.
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle
import re
from collections import namedtuple

from infer.classify import MultinomialNB
from infer.experiment import Experiment
from infer.nlp import FeatureExtractor

from infertweet.corpus.semeval import task_b_generator
from infertweet.corpus.twitter_corpus import tweet_generator
from infertweet.sentiment.experiment import run_experiment
from infertweet.sentiment.plot import start_plot
from infertweet.sentiment.constants import (
    TITLES, DEVELOPMENT, TRAINING, CHUNK_SIZE, FIRST_CHUNK)


Pickled = namedtuple('Pickled', 'extractor classifier')


class TestSemEval(Experiment):
    def _test_data(self):
        with open(DEVELOPMENT) as f:
            for instance in task_b_generator(f):
                yield instance


class TestSemEvalTrainingSet(Experiment):
    def _test_data(self):
        with open(TRAINING) as f:
            for instance in task_b_generator(f):
                yield instance


class TestTwitterCorpus(Experiment):
    def _test_data(self):
        import codecs
        with codecs.open(r"R:\_Other\Twitter\TwitterCorpus\test-corpus.csv", encoding='utf-8') as f:
            for line in f:
                line = line.split(',', 5)
                if len(line) != 5:
                    continue
                sentiment = line[1][1:-1].encode('utf-8')
                if sentiment not in ['positive', 'negative', 'neutral']:
                    continue
                text = line[4].encode('utf-8')
                # Strip surrounding '"'
                text = text[1:-1]
                yield self.DataInstance(text, sentiment)


class TrainSemEval(Experiment):
    def _train_data(self):
        with open(TRAINING) as f:
            for instance in task_b_generator(f):
                yield instance


class TrainSemEvalBoosting(TrainSemEval):  # TODO(bwbaugh): Is "boosting" right?
    def _train_data(self):
        for count, x in enumerate(super(TrainSemEvalBoosting, self)
                                  ._train_data(), start=1):
            yield x

        self.chunk_size /= 5
        while 1:
            with open(TRAINING) as f:
                old_count = count
                for tweet in task_b_generator(f):
                    document = tweet.text
                    features = self.extractor.extract(document)
                    if tweet.label != self._predict(features)[0]:
                        yield self.DataInstance(tweet.text, tweet.label)
                        count += 1
                if count < (old_count + 10):
                    break


class TrainSemEvalSelfLearning(TrainSemEval):
    def _train_data(self):
        for count, x in enumerate(super(TrainSemEvalSelfLearning, self)
                                  ._train_data(), start=1):
            yield x

        # for count, tweet in enumerate(tweet_generator(r"R:\_Other\Twitter\TwitterCorpus\twitter-sentiment_preprocess_1097409.json.bz2"), start=count + 1):
        # for tweet in tweet_generator(r"D:\Twitter\tweet_loc_denton-16.1302200911.json"):
        for tweet in tweet_generator(r"R:\_Other\Twitter\TwitterCorpus\twitter-sentiment_preprocess_1097409.json.bz2"):
            features = self.extractor.extract(tweet.text)
            label = None
            label, probability, = self._predict(features)
            # if tweet.text.count(':(') >= tweet.text.count(':)'):
            #     label, probability = 'negative', 1.0
            # elif ':)' in tweet.text:
            #     label, probability = 'positive', 1.0
            # if not label:
            #     continue
            # label = predict_func(features)

            if label == 'neutral':
                continue

            if probability > 0.8:
                count += 1
                yield self.DataInstance(tweet.text, label)
                # if count == 50000:
                #     worst = self.get_wrong_results(extractor, predict_func,
                #                                    correct_labels,
                #                                    test_data)
                #     pprint(worst)
                #     break

                # if count == 200000:
                #     break


class TrainStanford(Experiment):
    def _train_data(self):
        with open(r"R:\_Other\Twitter\2013\stanford\training.1600000.processed.noemoticon.shuffled.csv") as f:
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


class TrainSemEvalWithStanford(TrainSemEval, TrainStanford):
    def _train_data(self):
        for x in TrainSemEval._train_data(self):
            yield x
        for x in TrainStanford._train_data(self):
            yield x


class TrainZipStanfordWikipedia(TrainSemEval, TrainStanford, TrainWikipedia):
    def _train_data(self):
        for x in TrainSemEval._train_data(self):
            yield x
        stanford = TrainStanford._train_data(self)
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


def main():
    plot_queue = multiprocessing.Queue()
    confusion_queue = multiprocessing.Queue()
    start_plot(plot_queue, confusion_queue)

    first = SingleClassifier, TrainSemEvalSelfLearning, TestSemEval
    second = HierarchicalClassifier, TrainSemEvalSelfLearning, TestSemEval

    extractor = FeatureExtractor(tokenizer=tokenizer)
    extractor.min_n, extractor.max_n = 1, 2

    experiment = run_experiment(first, second, extractor, CHUNK_SIZE, FIRST_CHUNK)

    try:
        for data in experiment:
            data[TITLES[0]] = parse_performance(data[TITLES[0]])
            data[TITLES[1]] = parse_performance(data[TITLES[1]])
            plot_queue.put(data)
            confusion_queue.put(data)
            print data[TITLES[0]]['count'], data[TITLES[0]]['SemEval'], data[TITLES[1]]['SemEval'], data[TITLES[0]]['vocab'], data[TITLES[1]]['vocab']
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
