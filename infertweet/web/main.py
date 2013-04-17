# Copyright (C) 2013 Brian Wesley Baugh
"""Web interface allowing users to submit queries and get a response."""
import os
import collections
import colorsys
import datetime
import json
import logging
import socket
import subprocess
import threading

import rpyc
import tornado.ioloop
import tornado.web
import tornado.httpserver

from infertweet.config import get_config
from infertweet.twitter import Twitter


class MainHandler(tornado.web.RequestHandler):
    """Handles requests for the query input page."""

    def initialize(self):
        self.git_version = self.application.settings.get('git_version')

    def head(self, *args):
        """Handle HEAD requests by sending an identical GET response."""
        self.get(*args)

    def get(self):
        """Renders the query input page."""
        self.render("index.html", git_version=self.git_version)


class SentimentRequestHandler(MainHandler):
    """Base class for sentimen request handlers."""

    def initialize(self):
        self.git_version = self.application.settings.get('git_version')
        self.twitter = self.application.settings.get('twitter')
        self.rpc = self.application.settings.get('rpc_server')
        self.extract = self.rpc.root.extract
        self.subjective_classify = self.rpc.root.subjective_classify
        self.subjective_conditional = self.rpc.root.subjective_conditional
        self.polarity_classify = self.rpc.root.polarity_classify
        self.polarity_conditional = self.rpc.root.polarity_conditional

    def predict(self, features):
        """Use the classifier to predict sentiment."""
        label, probability = self.subjective_classify(features)
        if label == 'neutral':
            return label, probability
        else:
            return self.polarity_classify(features)

    def conditional(self, feature):
        """Get the contribution of an individual feature."""
        probability = self.subjective_conditional(feature, 'neutral')
        total = probability + self.subjective_conditional(feature, 'subjective')
        probability /= total
        if probability >= 0.5:
            return 'neutral', probability
        else:
            label, probability = self.polarity_classify(feature)
            probability = self.polarity_conditional(feature, 'positive')
            total = probability + self.polarity_conditional(feature, 'negative')
            probability /= total
            if probability < 0.5:
                return 'negative', 1 - probability
            else:
                return 'positive', probability

    def get_conditionals(self, features):
        predicted_features = []
        for feature in features:
            # We're just interested in unigrams.
            if len(feature) > 1:
                continue
            feature, = feature  # Unpack unit-sized tuple.
            label, probability = self.conditional((feature, ))
            predicted_features.append((feature, label, probability))
        return predicted_features

    def process_query(self, query):
        features = self.extract(query)
        label, probability = self.predict(features)
        features = self.get_conditionals(features)
        return query, features, label, probability


class SentimentQueryHandler(SentimentRequestHandler):
    """Handles sentiment queries and displays response."""

    def initialize(self):
        super(SentimentQueryHandler, self).initialize()
        self.twitter_cache = self.application.settings.get('twitter_cache')
        self.twitter_cache_seconds = self.application.settings.get(
            'twitter_cache_seconds')

    @tornado.web.asynchronous
    def get(self):
        """Handles GET sentiment query requests.

        GET Parameters:
            q: String of either the keywords to use for retrieving
                matching tweets from Twitter, or the text of a single
                document (tweet) to be classified. To be considered as
                keywords, the query must be 4 words or less and must be
                less than 30 characters, otherwise the string is assumed
                to be a single document.
            count: The number of tweets to request from Twitter that
                match the keywords in `q`. If specified, the
                `q`-parameter is forced to be interpreted as keywords.
                Maximum value is 100, defaults to 50.
        """
        self.query = self.get_argument('q')
        self.count = self.get_argument('count', default=None)
        if self.count or (len(self.query.split()) <= 4 and
                          len(self.query) <= 30):
            self.tweets = True
            if self.count is None:
                self.count = 50
            threading.Thread(target=self._twitter_search).start()
        else:
            self.tweets = False
            result = self.process_query(self.query)
            self._on_results([result])

    def _twitter_cache_update(self):
        """Remove items from the cache that have expired."""
        for query in self.twitter_cache.keys():
            cache_time, twitter_results = self.twitter_cache[query]
            now = datetime.datetime.now()
            if (now - cache_time >
                    datetime.timedelta(seconds=self.twitter_cache_seconds)):
                del self.twitter_cache[query]

    def _twitter_search(self):
        """Get matching tweets from Twitter."""
        self._twitter_cache_update()
        if self.query in self.twitter_cache:
            cache_time, twitter_results = self.twitter_cache[self.query]
        else:
            twitter_results = self.twitter.search(q=self.query,
                                                  rpp=self.count,
                                                  result_type='recent',
                                                  lang='en')
            cache_time = datetime.datetime.now()
            self.twitter_cache[self.query] = (cache_time, twitter_results)
        self._process_twitter(twitter_results)

    def _process_twitter(self, twitter_results):
        """Classify each tweet returned by Twitter."""
        results = []
        for tweet in twitter_results:
            result = self.process_query(tweet.text)
            result = (tweet, ) + result[1:]
            results.append(result)
        self._on_results(results)

    def _on_results(self, results):
        """Display the final results to the user.

        Args:
            results: List of tuples. If self.tweets, then the tuple
            consists of (tweet, features, label, probability), where
            `tweet` is a Tweepy `SearchResult` object, otherwise the
            `tweet` is replaced with a simple `text` string.
        """
        self.render("sentiment.html",
                    query=self.query,
                    results=results,
                    tweets=self.tweets,
                    overall_count=collections.Counter(x[2] for x in results),
                    color_code=color_code,
                    git_version=self.git_version)


class SentimentMisclassifiedHandler(SentimentRequestHandler):
    """Handles sentiment misclassification reports."""

    def post(self):
        """Handles POST sentiment misclassification requests.

        POST Parameters:
            text: String of the text that was misclassified.
            flag: String of the reported correct class label.
        """
        text = self.get_argument('text')
        flag = self.get_argument('flag').lower()

        # Classify the text to get the currently assigned label.
        text, features, label, probability = self.process_query(text)
        if flag == label:  # A valid flag can't be the current label.
            raise tornado.web.HTTPError(409)  # 409 Conflict

        self.render("misclassified.html",
                    text=text,
                    flag=flag,
                    color_code=color_code,
                    git_version=self.git_version)

        logger = logging.getLogger('ui.web.sentiment.misclassified')
        logger.info('\t'.join([flag, label, text]))


class SentimentAPIHandler(SentimentRequestHandler):
    """Handles sentiment API requests.

    Resource URL:
        http://.../api/sentiment/service_name.json
    """

    def get(self, service):
        """Handles GET API requests.

        Args:
            service: The name of the GET-service requested. Currently
                the only valid GET-service name is 'classify', which
                will classify a single document (tweet).

        GET Parameters:
            text: String representing the document to be classified.

        Returns:
            JSON dictionary of the classification results.

            Fields:
                text: String of the original input text.
                label: String of the sentiment classification label.
                confidence: Float of the confidence in the label.

            For example:

            {
                "text": "Today is March 30, 2013.",
                "confidence": 0.9876479882432573,
                "label": "neutral"
            }

        Raises:
            tornado.web.HTTPError: If the `service` name is unexpected.
        """
        if service != 'classify':
            raise tornado.web.HTTPError(404)

        self.set_header('Content-Type', 'application/json')

        text = self.get_argument('text')
        text, features, label, confidence = self.process_query(text)
        result = {'text': text, 'label': label, 'confidence': confidence}
        self.write(json.dumps(result))


def color_code(label, probability):
    """Converts float [0.0 - 1.0] to HTML color code."""
    if label == 'neutral':
        # We want the the neutral text to be darker (closer to black)
        # the more confident we are that it is neutral.
        rgb = (1 - probability, ) * 3  # all same causes grayscale.
    else:
        if label == 'negative':
            probability = 1 - probability
        rgb = colorsys.hsv_to_rgb(probability / 3, 1, 0.75)
    code = '#' + ''.join([hex(int(x * 256))[2:].zfill(2) for x in rgb])
    return code


def get_rpc_server(config):
    rpc_host = config.get('sentiment', 'rpc_host')
    rpc_port = config.getint('sentiment', 'rpc_port')
    rpc_server = rpyc.connect(rpc_host, rpc_port)
    return rpc_server


def get_git_version():
    """Get the SHA of the current Git commit.

    Returns:
        String tuple of the `git_version` as returned by `git describe`,
        and the `git_commit`, which is the full SHA of the last commit.
        If there was an error retrieving the values then (None, None) is
        returned.
    """
    try:
        git_commit = subprocess.check_output([
            'git', 'rev-parse', '--verify', 'HEAD']).rstrip()
        git_version = subprocess.check_output(['git', 'describe']).rstrip()
    except OSError, subprocess.CalledProcessError:
        git_version, git_commit = None, None
    return git_version, git_commit


def start_server(config, twitter, git_version):
    application = tornado.web.Application(
        [(r"/", MainHandler),
         (r"/sentiment/", SentimentQueryHandler),
         (r"/sentiment/misclassified", SentimentMisclassifiedHandler),
         (r"/api/sentiment/([^/.]+).json", SentimentAPIHandler)],
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static'),
        gzip=config.getboolean('web', 'gzip'),
        debug=config.getboolean('web', 'debug'),
        twitter=twitter,
        twitter_cache=dict(),
        twitter_cache_seconds=config.getint('web', 'twitter_cache_seconds'),
        rpc_server=get_rpc_server(config),
        git_version=git_version)
    http_server = tornado.httpserver.HTTPServer(application, xheaders=True)
    http_server.listen(config.getint('web', 'port'))

    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass


def setup_logging(config):
    if config.getboolean('web', 'debug'):
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger('')  # Root logger.
    logger.setLevel(log_level)

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
        datefmt='%m-%d %H:%M:%S')
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    web_query_log = config.get('sentiment', 'web_query_log')
    # Add the system's hostname before the file extension.
    web_query_log = (web_query_log[:-3] + socket.gethostname() +
                     web_query_log[-4:])

    file_handler = logging.FileHandler(web_query_log)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)d\t%(levelname)s\t%(name)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def main():
    """Starts the web server as a user interface to the system."""
    config = get_config()

    setup_logging(config)

    logger = logging.getLogger('ui.web')

    git_version, git_commit = get_git_version()
    if git_version:
        logger.info('Version: {0} ({1})'.format(git_version, git_commit))
    else:
        logger.warning('Could not detect current Git commit.')

    twitter = Twitter(config=config)

    logger.info('Starting web server on port {}'.format(config.getint('web',
                                                                      'port')))
    start_server(config=config, twitter=twitter,
                 git_version=(git_version, git_commit))

if __name__ == '__main__':
    main()
