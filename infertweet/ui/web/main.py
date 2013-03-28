# Copyright (C) 2013 Brian Wesley Baugh
"""Web interface allowing users to submit queries and get a response."""
import os
import colorsys
import subprocess
from datetime import datetime

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

    def get(self):
        """Renders the query input page."""
        self.render("index.html", git_version=self.git_version)


class SentimentQueryHandler(tornado.web.RequestHandler):
    """Handles sentiment queries and displays response."""

    def initialize(self):
        self.git_version = self.application.settings.get('git_version')
        self.web_query_log = self.application.settings.get('web_query_log')
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
        if probability < 0.5:
            return 'neutral', 1 - probability
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

    def log_query(self, label, probability, query):
        """Log the query to a file."""
        try:
            client_ip = self.request.headers['X-Real-Ip']
        except KeyError:
            client_ip = self.request.remote_ip

        message = '\t'.join([
            str(datetime.utcnow()) + ' +0000',
            client_ip,
            label,
            str(probability),
            query.encode('utf-8')])
        print message
        try:
            with open(self.web_query_log, mode='a') as f:
                f.write(message + '\n')
        except:
            pass

    def process_query(self, query):
        features = self.extract(query)
        label, probability = self.predict(features)
        features = self.get_conditionals(features)
        return features, label, probability

    def get(self):
        query = self.get_argument('q')
        texts, features_list, labels, probabilities = [], [], [], []
        if len(query.split()) <= 3 and len(query) <= 30:
            results = self.twitter.search(q=query, lang='en', rpp=50)
            for tweet in results:
                texts.append(tweet.text)
                features, label, probability = self.process_query(tweet.text)
                features_list.append(features)
                labels.append(label)
                probabilities.append(probability)
        else:
            texts.append(query)
            features, label, probability = self.process_query(query)
            features_list.append(features)
            labels.append(label)
            probabilities.append(probability)
        self.render("sentiment.html",
                    query=query,
                    texts=texts,
                    labels=labels,
                    probabilities=probabilities,
                    color_code=color_code,
                    features_list=features_list,
                    git_version=self.git_version)
        self.log_query(label, probability, query)


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
         (r"/sentiment/", SentimentQueryHandler)],
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static'),
        gzip=config.getboolean('web', 'gzip'),
        debug=config.getboolean('web', 'debug'),
        web_query_log=config.get('sentiment', 'web_query_log'),
        twitter=twitter,
        rpc_server=get_rpc_server(config),
        git_version=git_version)
    http_server = tornado.httpserver.HTTPServer(application, xheaders=True)
    http_server.listen(config.getint('web', 'port'))
    tornado.ioloop.IOLoop.instance().start()


def main():
    """Starts the web server as a user interface to the system."""
    config = get_config()
    git_version, git_commit = get_git_version()
    if git_version:
        print 'Version: {0} ({1})'.format(git_version, git_commit)
    else:
        print 'Could not detect current Git commit.'
    twitter = Twitter(config=config)
    print 'Starting web server on port {}'.format(config.getint('web', 'port'))
    start_server(config=config, twitter=twitter,
                 git_version=(git_version, git_commit))

if __name__ == '__main__':
    main()
