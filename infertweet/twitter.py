# Copyright (C) 2013 Wesley Baugh
"""Tools needed to interact with Twitter."""
import Queue
from tweepy.api import API as _API
from tweepy.models import Status
from tweepy.streaming import StreamListener
from tweepy.utils import import_simplejson
json = import_simplejson()


API = _API()


class UnroutableError(ValueError):
    pass


class QueueListener(StreamListener):
    """Tweets received from the stream are stored in an internal queue.
    Attributes:
        queue: Queue object that incoming tweets put into.
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
        """Iterate over the tweets in the queue."""
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
        return self.queue.qsize()


class Tweet(Status):
    @classmethod
    def parse(cls, data, api=API):
        """Parse a JSON string into a Tweet / Status object."""
        return Status.parse(api, json.loads(data))
