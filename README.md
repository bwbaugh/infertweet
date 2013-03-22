InferTweet
==========

Infer information from Tweets. Useful for human-centered computing
tasks, such as sentiment analysis, location prediction, authorship
profiling and more!

[![Build Status](https://travis-ci.org/bwbaugh/infertweet.png?branch=master)](https://travis-ci.org/bwbaugh/infertweet)

Sentiment Analysis
------------------

We provide three-class (positive, negative, objective-OR-neutral)
sentiment analysis on tweets.

Experiments are ongoing, but currently the system uses a hierarchical
classifier that first determines if a tweet is objective or subjective
(subjectivity classifier), and then if subjective determine if the tweet
is positive or negative (polarity classifier).

We use approximately 8,750 labeled training instances provided by the
[Sentiment Analysis in Twitter](http://www.cs.york.ac.uk/semeval-2013/task2/)
task for SemEval-2013. We then "freeze" the subjectivity classifier, as
we currently haven't been able to incorporate additional high quality
labeled or unlabeled objective-OR-neutral tweets or text. However, we
continue to train the polarity classifier through self-training on
approximately 1 million unlabeled tweets that are likely to contain
sentiment. The additional tweets were captured from Twitter if they had
a matching emoticon present in the text of the tweet.

At the time of this writing, we are currently awaiting the results of
our system in the SemEval-2013 competition. In the mean time, we have a
lot more experimental ideas that may improve the performance of our
classifiers!

### RPC server

The sentiment analysis classifier can be loaded from file and served
using a RPC server. This allows the classifier to potentially be used by
many applications, as well as being able to stay loaded even if another
application that depends on the classifier needs to restart or update.

### Web user interface

We have added a very simple web interface that allows users to query the
system. Lots of upcoming features are planned for the web interface.
