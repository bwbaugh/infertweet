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

### SemEval-2013

An early version of our system was entered in the SemEval-2013
competition. Our simple system (Naive Bayes with unigrams + bigrams)
scored 25th out of 48 submissions, which while not state-of-the-art is
still not too bad.

The evaluation metric was the average F-measure of the positive and
negative classes. Our system achieved an F-measure of `0.5437`, while
the top system achieved `0.6902`.

#### Results of system for SemEval-2013

    Confusion table:
    gs \ pred| positive| negative|  neutral
    ---------------------------------------
     positive|      841|      233|      498
     negative|       74|      324|      203
      neutral|      276|      196|     1168


    Scores:
    class                    prec                 recall     fscore
    positive      (841/1191) 0.7061    (841/1572) 0.5350     0.6088
    negative       (324/753) 0.4303     (324/601) 0.5391     0.4786
    neutral      (1168/1869) 0.6249   (1168/1640) 0.7122     0.6657
    --------------------------------------------------------------------
    average(pos and neg)                                          0.5437

In the mean time, we have a lot more experimental ideas that may improve
the performance of our classifier, so it's time to get experimenting!

### RPC server

The sentiment analysis classifier can be loaded from file and served
using a RPC server. This allows the classifier to potentially be used by
many applications, as well as being able to stay loaded even if another
application that depends on the classifier needs to restart or update.

### Web user interface

We have added a very simple web interface that allows users to query the
system. Lots of upcoming features are planned for the web interface.

### RESTful JSON API

#### GET sentiment/classify

##### Resource URL

http://.../api/sentiment/classify.json

##### Parameters

- text: String representing the document to be classified.

##### Response object fields

- text: String of the original input text.
- label: String of the sentiment classification label.
- confidence: Float of the confidence in the label.

##### Example request

GET `http://.../api/sentiment/classify.json?text=Today+is+March+30%2C+2013.`

    {
        "text": "Today is March 30, 2013.",
        "confidence": 0.9876479882432573,
        "label": "neutral"
    }
