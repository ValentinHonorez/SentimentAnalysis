#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData

from sqlalchemy import insert, select
import json 
import tweepy
import API_and_Database_function as func




ACCESS_TOKEN = 'XXX'
ACCESS_SECRET = 'XXX'
CONSUMER_KEY = 'XXX'
CONSUMER_SECRET = 'XXX'
api = func.Tweepy_Access(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)



search_words = "#politique"
date_since = "2018-11-16"
language = "fr"
number_items =  500
tweets = func.Search_Tweets(api, search_words, date_since, language , number_items)
connection, Twitter_Sentiment_Analysis = func.Database_Acces("mysql://root@localhost/sentiment?charset=utf8mb4", 'utf8', "Twitter_Sentiment_Analysis4")




# Build a list of dictionaries: values_list
values_list = {}



for tweet in tweets:

    re_tweet, retweet_TweetID, user_retweet, userID_retweet, user_Screen_retweet = func.get_Retweets(tweet, api)
    Hashtag = func.get_Hashtags(tweet)
    symbols = func.get_Symbols(tweet)
    city, country = func.get_Location(tweet)
    mention = func.get_mention(tweet)
    url = func.get_url(tweet)
    image, video = func.get_media(tweet)
    reply, a,b = func.get_Reply(tweet)

    

 


    values_list.update({'Tweets_ID': tweet.id, 'Tweets_data': tweet.text.encode('utf8'), 'Tweets_Date_of_creation': tweet.created_at, 'Retweet': re_tweet , 'Retweet_original_TweetID' : retweet_TweetID , 'Retweet_original_authorID': userID_retweet , 'Retweet_original_author_Name' : user_retweet, 'Retweet_original_author_Screen_Name' : user_Screen_retweet,


          'Reply' : reply , 'Reply_original_authorID' : tweet.in_reply_to_user_id ,  'Reply_original_author_Screen_Name' : tweet.in_reply_to_screen_name , 
    'User_ID' : tweet.user.id , 'USERNAME' : tweet.user.name.encode('utf8'), 'SCREEN_NAME' : tweet.user.screen_name.encode('utf8') , 'City_Name' : city , 'Country_Name' : country , 
    'Language' : tweet.lang.encode('utf8') , 'Verified_account' : tweet.user.verified , 
    'Numbers_of_followers' : tweet.user.followers_count, 'Numbers_of_friends' : tweet.user.friends_count , 'User_mentions' : mention,
    'Hashtags' : Hashtag , 'Other_symbols' : symbols , 'Video' : video , 'Image' : image, 'URLS' : url} )
    

    # Build an insert statement for the data table: stmt
    stmt = insert(Twitter_Sentiment_Analysis)

    # Execute stmt with the values_list: results
    results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)

