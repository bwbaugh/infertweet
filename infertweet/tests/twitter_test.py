# Copyright (C) 2013 Wesley Baugh
from nose.tools import assert_raises
from infertweet.twitter import Tweet, QueueListener, UnroutableError


EXAMPLE_TWEETS = r'''
{"retweet_count":0,"in_reply_to_user_id":81692398,"text":"@__Hermosaa :( Whats wrong Tea?","in_reply_to_status_id_str":"257649019256446976","favorited":false,"id_str":"257649192485388289","coordinates":null,"created_at":"Mon Oct 15 01:08:49 +0000 2012","in_reply_to_user_id_str":"81692398","geo":null,"truncated":false,"source":"web","entities":{"urls":[],"hashtags":[],"user_mentions":[{"indices":[0,11],"id_str":"81692398","screen_name":"__Hermosaa","name":"\u0442\u043d\u03b1\u0442\u0455 \u0442\u043c\u03c3\u03b7\u0454\u0443\u2665","id":81692398}]},"contributors":null,"place":null,"retweeted":false,"user":{"profile_background_tile":true,"friends_count":134,"is_translator":false,"default_profile":false,"profile_background_image_url_https":"https:\/\/si0.twimg.com\/profile_background_images\/492442329\/jada3.jpg","profile_sidebar_fill_color":"DDEEF6","followers_count":140,"id_str":"544608686","notifications":null,"created_at":"Tue Apr 03 21:34:04 +0000 2012","profile_sidebar_border_color":"000","url":"http:\/\/www.instagr.am\/_jadadenise","description":"you know my name. not my story and if a hoe want war Imma be the last bitch standing ...  PROMISED! \u270c","geo_enabled":true,"profile_image_url_https":"https:\/\/si0.twimg.com\/profile_images\/2693864669\/ce9ce11d320eb02c8a1cb99b7377addc_normal.jpeg","default_profile_image":false,"lang":"en","profile_use_background_image":true,"profile_image_url":"http:\/\/a0.twimg.com\/profile_images\/2693864669\/ce9ce11d320eb02c8a1cb99b7377addc_normal.jpeg","statuses_count":3328,"verified":false,"favourites_count":13,"profile_text_color":"333333","location":"Skirtt Skirtt","screen_name":"__TrillAssJadaa","profile_background_image_url":"http:\/\/a0.twimg.com\/profile_background_images\/492442329\/jada3.jpg","protected":false,"contributors_enabled":false,"following":null,"time_zone":"Arizona","profile_link_color":"9C12A3","name":"Pink Thugg.  \ue003","id":544608686,"listed_count":0,"follow_request_sent":null,"utc_offset":-25200,"profile_background_color":"000000"},"in_reply_to_screen_name":"__Hermosaa","in_reply_to_status_id":257649019256446976,"id":257649192485388289}
{"retweet_count":0,"in_reply_to_user_id":null,"text":"Kind of want chili cheese fries :)","in_reply_to_status_id_str":null,"favorited":false,"id_str":"257649192774811649","coordinates":null,"created_at":"Mon Oct 15 01:08:49 +0000 2012","in_reply_to_user_id_str":null,"geo":null,"truncated":false,"source":"\u003Ca href=\"http:\/\/twitter.com\/download\/iphone\" rel=\"nofollow\"\u003ETwitter for iPhone\u003C\/a\u003E","entities":{"urls":[],"hashtags":[],"user_mentions":[]},"contributors":null,"place":null,"retweeted":false,"user":{"profile_background_tile":false,"friends_count":170,"is_translator":false,"default_profile":false,"profile_background_image_url_https":"https:\/\/si0.twimg.com\/images\/themes\/theme17\/bg.gif","profile_sidebar_fill_color":"E6F6F9","followers_count":141,"id_str":"261580681","notifications":null,"created_at":"Sun Mar 06 07:12:37 +0000 2011","profile_sidebar_border_color":"DBE9ED","url":null,"description":"Im 19, :) im sorry that people are so jealous of me, but i cant help it that im so popular.","geo_enabled":true,"profile_image_url_https":"https:\/\/si0.twimg.com\/profile_images\/2715915142\/03de4f812db7d4946f666d200b02b055_normal.jpeg","default_profile_image":false,"lang":"en","profile_use_background_image":true,"profile_image_url":"http:\/\/a0.twimg.com\/profile_images\/2715915142\/03de4f812db7d4946f666d200b02b055_normal.jpeg","profile_banner_url":"https:\/\/si0.twimg.com\/profile_banners\/261580681\/1348631908","statuses_count":9458,"verified":false,"favourites_count":26,"profile_text_color":"333333","location":"Lancaster","screen_name":"i_be_rosa","profile_background_image_url":"http:\/\/a0.twimg.com\/images\/themes\/theme17\/bg.gif","protected":false,"contributors_enabled":false,"following":null,"time_zone":"Pacific Time (US & Canada)","profile_link_color":"CC3366","name":"Rosa Garcia","id":261580681,"listed_count":0,"follow_request_sent":null,"utc_offset":-28800,"profile_background_color":"DBE9ED"},"in_reply_to_screen_name":null,"in_reply_to_status_id":null,"id":257649192774811649}
                  '''
EXAMPLE_TWEETS = EXAMPLE_TWEETS.strip().split('\n')


class TestQueueListener(object):
    def setup(self):
        self.stream = QueueListener()
        for tweet in EXAMPLE_TWEETS:
            self.stream.on_data(tweet)

    def test_iter(self):
        tweets = iter(self.stream)
        next(tweets)

    def test_parse_tweets(self):
        for data in self.stream:
            tweet = Tweet.parse(data)
            assert tweet.text

    def test_length(self):
        assert len(self.stream) == 2

    def test_on_data_limit(self):
        self.stream.on_data('{"limit":{"track":65}}')
        assert self.stream.limit_track == 65

    def test_on_data_unroutable(self):
        assert_raises(UnroutableError, self.stream.on_data, '{"none": null}')

    def test_maxsize(self):
        self.stream = QueueListener(maxsize=1)
        assert self.stream.on_data(EXAMPLE_TWEETS[0])
        assert not self.stream.on_data(EXAMPLE_TWEETS[0])
