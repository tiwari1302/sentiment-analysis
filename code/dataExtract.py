#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tweepy
from textblob import TextBlob
import string


# In[2]:

#These key need to be private  

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token = (access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[ ]:


# # # THIS IS THE SEARCH QUERY # # #
#taglist = ['#lockdownindia', '#lockdownextension', '#covid19india']
taglist = ['#IndiaChinaBorderTension', '#IndiaChinaFaceOff', '#IndiaChinaBorder', '#IndiaChinaFaceOffNow', '#IndiaWillPunishChina']
max_tweets = 10000 #each tag will extract this number of tweets. So total number of tweets are: max_tweets*3 
data = []

for p, tag in enumerate(taglist):
    tweets = tweepy.Cursor(api.search, q=tag,lang="en").items(max_tweets)
    for i, tweet in enumerate(tweets):
        if i%100== 0:
            print("Number of preprocessed Tweets :", i+((p)*max_tweets)+1)
            col_Names=["tweet_id", "date", "time", "user_name", "tweet", "loaction"]
            data_df = pd.DataFrame.from_records(data, columns=col_Names)
            data_df.to_csv("tweet_data.csv")
            print("Total number of collected tweets so for:", len(data)) 
            
        if tweet.id_str!= None and tweet.created_at != None and tweet.user.screen_name != None and tweet.text != None and tweet.user.location != None and (tweet.user.location.find('india') != -1 or tweet.user.location.find('India') != -1): #and tweet.user.location.strip() != 'india' \
        #and tweet.user.location.strip() != 'India':
            temp_list = []
            temp_list.append(tweet.id_str)
            temp_list.append(str(tweet.created_at).strip().split(' ')[0])
            temp_list.append(str(tweet.created_at).strip().split(' ')[1])
            temp_list.append(tweet.user.screen_name)
            temp_list.append(tweet.text)
            temp_list.append(tweet.user.location)
            data.append(temp_list)
print("Total Number of Collected Tweets:", len(data)) 


# In[ ]:


col_Names=["tweet_id", "date", "time", "user_name", "tweet", "loaction"]
data_df = pd.DataFrame.from_records(data, columns=col_Names)
data_df.to_csv("tweet_data.csv")

