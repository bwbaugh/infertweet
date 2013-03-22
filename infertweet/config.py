# Copyright (C) 2013 Wesley Baugh
"""Configuration file access and settings."""
import errno
import os
import sys
import ConfigParser


CONFIG_FNAME = 'infertweet.ini'


def create_default_config():
    """Create a default config file."""
    config = ConfigParser.SafeConfigParser()

    config.add_section('sentiment')
    config.set('sentiment', 'path', 'PATH/TO/CLASSIFIER/')
    config.set('sentiment', 'classifier', 'sentiment-classifier.pickle')
    config.set('sentiment', 'rpc_host', 'localhost')
    config.set('sentiment', 'rpc_port', '18861')
    config.set('sentiment', 'web_query_log', 'web_log_queries.txt')

    config.add_section('web')
    config.set('web', 'port', '8080')
    config.set('web', 'gzip', 'true')
    config.set('web', 'debug', 'true')

    return config


def get_config(fname=CONFIG_FNAME, create=True, exit=True):
    """Reads a configuration file from disk."""
    config = ConfigParser.SafeConfigParser()
    try:
        with open(fname) as f:
            config.readfp(f)  # pragma: no branch
    except IOError as e:
        if e.errno != errno.ENOENT:
            raise  # pragma: no cover
        if create:
            print 'Configuration file not found! Creating one...'
            config = create_default_config()
            with open(fname, mode='w') as f:
                config.write(f)
            message = 'Please edit the config file named "{}" in directory "{}"'
        else:
            message = 'Configuration file "{}" not found in directory "{}"'
        print message.format(CONFIG_FNAME, os.getcwd())
        if exit:
            sys.exit(errno.ENOENT)
        else:
            if not create:
                return None
    return config
