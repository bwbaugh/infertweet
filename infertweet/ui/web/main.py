# Copyright (C) 2013 Brian Wesley Baugh
"""Web interface allowing users to submit queries and get a response."""
import os
import colorsys

import rpyc
import tornado.ioloop
import tornado.web
import tornado.httpserver

from infertweet.config import get_config


class MainHandler(tornado.web.RequestHandler):
    """Handles requests for the query input page."""

    def get(self):
        """Renders the query input page."""
        self.render("index.html")


class SentimentQueryHandler(tornado.web.RequestHandler):
    """Handles sentiment queries and displays response."""

    def initialize(self):
        self.web_query_log = self.application.settings.get('web_query_log')
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

    def log_query(self, label, probability, query):
        """Log the query to a file."""
        try:
            client_ip = self.request.headers['X-Real-Ip']
        except KeyError:
            client_ip = self.request.remote_ip

        message = '{}\t{}\t{}\t{}'.format(
            client_ip,
            label,
            probability,
            query.encode('utf-8'))
        print message
        try:
            with open(self.web_query_log, mode='a') as f:
                f.write(message + '\n')
        except:
            pass

    def process_query(self, query):
        features = self.extract(query)
        label, probability = self.predict(features)
        predicted_features = []
        for feature in features:
            f_label, f_prob = self.conditional(feature)
            predicted_features.append((feature, f_label, f_prob))
        features = predicted_features
        return features, label, probability

    def get(self):
        query = self.get_argument('q')
        features, label, probability = self.process_query(query)
        self.render("sentiment.html",
                    query=query,
                    label=label,
                    probability=probability,
                    color_code=color_code,
                    features=features)
        self.log_query(label, probability, query)


def color_code(label, probability):
    """Converts float [0.0 - 1.0] to HTML color code."""
    if label == 'neutral':
        code = '#808080'
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


def start_server(config):
    application = tornado.web.Application(
        [(r"/", MainHandler),
         (r"/sentiment/", SentimentQueryHandler)],
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static'),
        gzip=config.getboolean('web', 'gzip'),
        debug=config.getboolean('web', 'debug'),
        web_query_log=config.get('sentiment', 'web_query_log'),
        rpc_server=get_rpc_server(config))
    http_server = tornado.httpserver.HTTPServer(application, xheaders=True)
    http_server.listen(config.getint('web', 'port'))
    tornado.ioloop.IOLoop.instance().start()


def main():
    """Starts the web server as a user interface to the system."""
    config = get_config()
    print 'Starting web server on port {}'.format(config.getint('web', 'port'))
    start_server(config)

if __name__ == '__main__':
    main()
