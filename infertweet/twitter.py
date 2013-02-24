# Copyright (C) 2013 Wesley Baugh
"""Tools needed to interact with Twitter."""
import Queue
import time
import socket
import httplib

from tweepy import OAuthHandler
from tweepy import Stream as TweepyStream
from tweepy.api import API as tweepy_api
from tweepy.models import Status
from tweepy.streaming import StreamListener
from tweepy.utils import import_simplejson
json = import_simplejson()


_API = tweepy_api()


class UnroutableError(ValueError):
    pass


class TwitterStreamError(Exception):
    pass


class QueueListener(StreamListener):
    """Tweets received from the stream are stored in an internal queue.

    Attributes:
        queue: Queue object that incoming tweets are put into.
        num_handled: Total number of incoming tweets handled by this
            listener.
        limit_track: Number of tweets that were NOT sent by Twitter that
            matched our filter.
    """
    def __init__(self, maxsize=0):
        """Creates a new stream listener with an internal queue for tweets."""
        super(QueueListener, self).__init__()
        self.queue = Queue.Queue(maxsize=maxsize)
        self.num_handled = 0
        self.limit_track = 0

    def on_data(self, data):
        """Routes the raw stream data to the appropriate method."""
        data_json = json.loads(data)
        if 'in_reply_to_status_id' in data_json:
            if self.on_status(data) is False:
                return False
        elif 'limit' in data_json:
            self.on_limit(data_json['limit']['track'])
        elif 'delete' in data_json:
            delete = data_json['delete']['status']
            self.on_delete(delete['id'], delete['user_id'])
        else:
            raise UnroutableError('JSON string: """{0}"""'.format(data))
        return True

    def on_status(self, data):
        """Puts each JSON string format tweet into the queue."""
        # Note that in this overridden method 'data' is a string whereas
        # in the default listener method this method is passed an object in the
        # format of: status = Status.parse(self.api, json.loads(data))
        try:
            self.queue.put(data, timeout=0.1)
        except Queue.Full:
            return False
        self.num_handled += 1

    def on_error(self, status_code):
        raise TwitterStreamError('ERROR: {0}'.format(status_code))

    def on_limit(self, track):
        """Called when a limitation notice arrvies.

        Limit notices indicate that additional tweets matched a filter,
        however they were above an artificial limit placed on the stream
        and were not sent. The value of `track` indicates how many
        tweets in total matched the filter but were not sent since the
        stream was opened.
        """
        self.limit_track = track

    def __iter__(self):
        """Iterate over the tweets currently in the queue."""
        while True:
            try:
                data = self.queue.get(timeout=0.1)
                # if data is None:
                #     self.queue.task_done()
                #     return
                self.queue.task_done()
                yield data
            except Queue.Empty:
                return

    def __len__(self):
        """Number of tweets in the queue."""
        return self.queue.qsize()


class Tweet(Status):
    @classmethod
    def parse(cls, data, api=_API):
        """Parse a JSON string into a Tweet / Status object."""
        return Status.parse(api, json.loads(data))


class Stream(object):
    """Allows use of the Twitter Streaming API.

    Constants:
        MAX_TCPIP_TIMEOUT: Maximum seconds to wait between retries for
            TCPIP errors.
        TCPIP_STEP: Additional seconds to wait between after additional
            TCPIP errors.
        MAX_HTTP_TIMEOUT: Maximum seconds to wait between retries for
            HTTP errors.
        HTTP_STEP: Additional seconds to wait between after additional
            HTTP errors.

    Attributes:
        tcpip_delay: Current number of seconds to wait after a TCPIP error.
        http_delay: Current number of seconds to wait after a HTTP error.
    """
    MAX_TCPIP_TIMEOUT = 16
    TCPIP_STEP = 0.25
    MAX_HTTP_TIMEOUT = 320
    HTTP_STEP = 5

    def __init__(self, listener, consumer_key=None, consumer_secret=None,
                 access_token=None, access_token_secret=None, config=None):
        """Creates a new Stream object.

        Args:
            listener: StreamListener object for putting the stream data.
            consumer_key: String for authenticating with Twitter.
            consumer_secret: String for authenticating with Twitter.
            access_token: String for authenticating with Twitter.
            access_token_secret: String for authenticating with Twitter.
            config: ConfigParser object with all of the above parameters
                under a 'twitter' section.
        """
        if config:
            consumer_key = config.get('twitter', 'consumer_key')
            consumer_secret = config.get('twitter', 'consumer_secret')
            access_token = config.get('twitter', 'access_token')
            access_token_secret = config.get('twitter', 'access_token_secret')
        elif not (consumer_key and consumer_secret and
                  access_token and access_token_secret):
            raise ValueError('Missing Twitter credentials')
        # Number of seconds to wait after an exception before restarting.
        self.tcpip_delay = self.TCPIP_STEP
        self.http_delay = self.HTTP_STEP
        # Setup stream
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self._stream = TweepyStream(auth, listener)

    def _run(self, endpoint, **kwargs):
        while True:
            try:
                self.running = True
                endpoint(**kwargs)  # Blocking!
            except KeyboardInterrupt:
                print 'KEYBOARD INTERRUPT'
                return
            except (socket.error, httplib.HTTPException):
                print ('TCP/IP Error: Restarting after '
                       '{0} seconds.'.format(self.tcpip_delay))
                time.sleep(min(self.tcpip_delay, self.MAX_TCPIP_TIMEOUT))
                self.tcpip_delay += 0.25
            except TwitterStreamError as e:
                print 'ERROR:', repr(e)
                return
            finally:
                print 'Disconnecting stream'
                self._stream.disconnect()

    def sample(self, **kwargs):
        """Connect the the sample endpoint, which uses no filtering."""
        self._run(self._stream.sample, **kwargs)

    def filter(self, follow=None, track=None, locations=None, **kwargs):
        """Connect to the filter endpoint.

        Args:
            follow: List of users to follow.
            track: List of keywords to match.
            locations: List of location bounding boxes to match.
        """
        self._run(self._stream.filter, follow=follow, track=track,
                  locations=locations, **kwargs)
