# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:24:20 2018

@author: micha
"""

from twython import Twython
from twython import TwythonStreamer
import collections
import simplejson as json

with open('jason/twitter_credentials.json', 'r') as file:  
    creds = json.load(file)

tweets=[]

class MyStreamer(TwythonStreamer):
    
    def on_success(self,data):
        tweets.append(data)
        print("received tweet", len(tweets))
        if(len(tweets)>2000):
            self.disconnect()
            
    def on_error(self, data):
        print(status_code, data)
        #self.disconnect()
        
        
stream = MyStreamer(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],creds['ACCESS_TOKEN'],creds['ACCESS_TOKEN_SECRET'])
stream.statuses.filter(track=['is'])


cleaned_tweets=[]
for tweet in tweets:
   if "entities" in tweet.keys():
       cleaned_tweets.append(tweet)
       
       
top_hashtags = collections.Counter(hashtag['text'].lower() for tweet in cleaned_tweets for hashtag
                       in tweet["entities"]["hashtags"])

print(top_hashtags.most_common(10))



#twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)
#
#for status in twitter.search(q= '"Angela Merkel"', lang='de')["statuses"]:
#    user = status["user"]["screen_name"].encode('utf-8')
#    text = status["text"].encode('utf-8')
#    print user, " : ", text
#    print