#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData

from sqlalchemy import insert, select, delete, update
import json 
import tweepy
import pandas as pd
import re



def Tweepy_Access(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET) : 
    """This function allows to connect to the API
    
    Args:
        ACCESS_TOKEN (str)
        ACCESS_SECRET (str)
        CONSUMER_KEY (str)
        CONSUMER_SECRET (str)
        
    
    Returns:
        API <class 'tweepy.api.API'> : enable to use the API
    """
    try : 
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    except Exception:
        print("Impossible to connect")
    # Create the api to connect to twitter with your creadentials
    try : 
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    except Exception:
        print("Impossible to go into the API")
    #---------------------------------------------------------------------------------------------------------------------
    # wait_on_rate_limit= True;  will make the api to automatically wait for rate limits to replenish
    # wait_on_rate_limit_notify= True;  will make the api  to print a notification when Tweepyis waiting for rate limits to replenish
    #---------------------------------------------------------------------------------------------------------------------
    return api

def Search_Tweets(api, search_words, date_since = "2018-11-16", language = "en", number_items = 1) :
    """This function is for searching tweets with specifical words/ langage / date
    
    Args:
        api <class 'tweepy.api.API'> : enable to use the tweepy API
        search_words (str): the specificals words/topic/#... that need to be in the tweets
        date_since (str, optional): from which date you want to take the tweets. Defaults to "2018-11-16".
        language (str, optional): which langage the tweets need to be. Defaults to "en".
        number_items (int, optional): How many tweets do you want. Defaults to 1.
    
    Returns:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...

    """
    
    tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang= language,
              since=date_since).items(number_items)

    
    return tweets


def Database_Acces(path, kind_encoding , Table_name ):
    """Create the connexion to the database on the server
    
    Args:
        path (str): see https://overiq.com/sqlalchemy-101/installing-sqlalchemy-and-connecting-to-database/ for further informations
        kind_encoding (str): the encoding used ('utf8' is the most used)
        Table_name (str): the name of the table in the server
    
    Returns:
        connection <class 'sqlalchemy.engine.base.Connection'>: will be used when you have to execute a statement to the database 
        The_Table_name <class 'sqlalchemy.sql.schema.Table'> :  variable attached to the table of the database 
    """
    engine = create_engine(path, echo = True, encoding = kind_encoding)
    metadata = MetaData()
    connection = engine.connect()
    The_Table_name = Table(Table_name, metadata, autoload= True, autoload_with= engine)
    return connection, The_Table_name


def get_Retweets(tweet, api  ) :
    """If the tweet is a retweet, it will contain a retweeted status, 
    this function will in this case get the id/name/screen name of the original tweet author and the id of original tweet
    
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
        api <class 'tweepy.api.API'> : enable to use the tweepy API
    Returns:
        re_tweet (int) : if it's a retweet or not 
        retweet_TweetID (int) : the id of the original tweet
        user_retweet (str) : the name of the user of the original tweet
        userID_retweet (int) : the id of the user of the original tweet
        user_Screen_retweet (str) : the screen name of the user of the original tweet
    """
    if hasattr(tweet, 'retweeted_status'):
         re_tweet = 1
         retweet_TweetID = tweet.retweeted_status.id_str
         tweets2 = api.get_status(tweet.retweeted_status.id_str)
         #print(tweets2.user.name.encode('utf8'))
         user_retweet = tweets2.user.name.encode('utf8')
         userID_retweet = tweets2.user.id
         user_Screen_retweet = tweets2.user.screen_name.encode('utf8')
    else:
         re_tweet = 0
         retweet_TweetID = 0
         user_retweet = 'None'
         userID_retweet = 0
         user_Screen_retweet = 'None'            

    return re_tweet, retweet_TweetID, user_retweet, userID_retweet, user_Screen_retweet

def get_Hashtags(tweet) :
    """ this function allows to know if the tweet is a retweet or not
    
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
    
    Returns:
        (int) : if there is an hashtag or not in the tweet
    """
    if  len(tweet.entities['hashtags']) == 0 :
        Hashtag = 0
    else :
        Hashtag = 1
    return Hashtag

def get_Symbols(tweet):
    """ this function allows to know if the tweet has specials symbols like cash-tag or not
    
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
    
    Returns:
        (int) : if there is special symbols or not in the tweet
    """
    if  len(tweet.entities['symbols']) == 0 :
        symbols = 0
    else :
        symbols = 1
    return symbols


def get_Location(tweet, coding = 'utf8') : 
    """ this function allows to know the location of the tweet if given (city and country)
    
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
        coding (str): the encoding used ('utf8' is the most used)
    Returns:
        city (str) : the tweet author's city
        country (str) : the tweet author's country
    """
    try : 
        location = tweet.user.location.encode(coding)
        location = location.replace("/", ",").split(',')
        if len (tweet.user.location) >= 2 :
            city = location[0]
            country = location[1].replace(" ", "", 1)
        city = location[0]
        country = location[1]
    except : 
        city = 'None'
        country = 'None'
    return city, country


def get_mention(tweet, coding = 'utf8') : 
    """ this function allows to know if given the user mentionned in the tweet
    
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
        coding (str): the encoding used ('utf8' is the most used)
    Returns:
        mention (str) : the user mentionned in the tweet
    """
    try:
        mention = tweet.entities['user_mentions'][0]["name"].encode(coding)
    except:
        mention = 'None'
    return mention


def get_url(tweet, coding = 'utf8') :
    """ this function allows to know if given the url of the tweet
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
        coding (str): the encoding used ('utf8' is the most used)
    Returns:
        url (str) : the url present in the tweet
    """ 
    try :
        url = tweet.entities['urls'][0]['url'].encode(coding)
    except :
        url = 'None'
    return url
    
def get_media(tweet, coding = 'utf8') :
    """ this function allows to know if given the picture and video of the tweet
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
        coding (str): the encoding used ('utf8' is the most used)
    Returns:
        image (str) : url of the picture(s) present in the tweet
        video (str) : url of the video(s) present in the tweet
    """
    try : 
        image = tweet.extended_entities['media'][0]['media_url_https'].encode(coding)
        video = tweet.extended_entities['media'][0]['video_info']['variants'][0]['url'].encode(coding)
    except: 
        image = 'None'
        video = 'None'
    return image, video

    
def get_Reply (tweet):
    """ this function allows to know if a tweet is a reply to a tweet and know the id/username/ screen name of the initial tweet
    Args:
        tweets <class 'tweepy.cursor.ItemIterator'>: contain all the features of a tweets. 
        Eg : the numbers of followers of the person of the tweet, if it's a verified account, the text of thet tweet...
    
    Returns:
        reply (int) : is it a reply or not
        Reply_ID (int) : the id of the initial author of the tweet
        Reply_Screen (str) : the screen name of the initial author of the tweet 
    """
    if  tweet.in_reply_to_status_id != None : #if you want user in_reply_to_user_id
        reply = 0
        Reply_ID = tweet.in_reply_to_user_id
        Reply_Screen = tweet.in_reply_to_screen_name
    else :
        reply = 1
        Reply_ID = tweet.in_reply_to_user_id
        Reply_Screen = tweet.in_reply_to_screen_name
    return reply, Reply_ID, Reply_Screen



def All_tweets_Delete(Name_Table, connection) : 
    """This function delete all the informations of a table of the database
    
    Args:
        connection <class 'sqlalchemy.engine.base.Connection'> : will be used when you have to execute a statement to the database 
        The_Table_name <class 'sqlalchemy.sql.schema.Table'> :  variable attached to the table of the database
    """
    delete_stmt = delete(Name_Table)

    # Execute the statement: results
    results = connection.execute(delete_stmt)

    # Print affected rowcount
    print(results.rowcount)

def export_to_csv(The_Table_name, connection, CSV_name):
    """This function exports in csv from the database the Tweets Data and the Tweets ID. It also adds some columns to prepare the manual annotation.
    
    Args:
        
        The_Table_name <class 'sqlalchemy.sql.schema.Table'> :  variable attached to the table of the database
        connection <class 'sqlalchemy.engine.base.Connection'> : will be used when you have to execute a statement to the database
        CSV_name (str) : Name of the csv file you want to create
    """
    stmt = select([The_Table_name.columns.Tweets_data])
    stmt2 = select([The_Table_name.columns.Tweets_ID])
    final_result = []
    final_result2 = []
    for results in connection.execute(stmt).fetchall() :
        results2 = list(results)
        final_result.append(results2)

    for results2 in connection.execute(stmt2).fetchall() :
        final_result2.append(results2)        

    zipped = list(zip(final_result2 , final_result))
    df = pd.DataFrame(zipped)
    list_labels = ['Tweets_ID', 'Tweets_data' ]
    df.columns = list_labels
    final2 = []
    try :
        for i in df['Tweets_data'] :
            for j in i :
                first = re.sub("[\r\n]+", " ", j)#remove the \n
                final = re.sub(" +", " ", first)#replace all the spaces by only 1 space
                final2.append(final)
    except : 
        pass

    zipped = list(zip(final_result2 , final2))
    df2 = pd.DataFrame(zipped)
    list_labels = ['Tweets_ID', 'Tweets_data' ]
    df2.columns = list_labels
    df2.loc[:,'Manual_Sentiment_Annotation'] = df2.sum(numeric_only=True, axis=1)
    df2.loc[:,'User_of_the_annotation'] = df2.sum(numeric_only=True, axis=1)
    df2.to_csv(CSV_name, encoding = 'utf16', index = False, sep = ';')


def update_annotations_db(The_Table_name, connection, Dataframe_from_CSV) :
    Sentiment_df = pd.read_csv(Dataframe_from_CSV, encoding = 'UTF-16 LE',  sep = '\t')

    for i in range(len(Sentiment_df['Tweets_ID'])) :
        update_stmt = update(The_Table_name).values(Manual_Sentiment_Annotation= Sentiment_df['Manual_Sentiment_Annotation'][i])
        update_stmt2 = update_stmt.where(The_Table_name.columns.Tweets_ID == int(Sentiment_df['Tweets_ID'][i][1:-1]))
        update_results = connection.execute(update_stmt2)
    
        update_stmt_1 = update(The_Table_name).values(User_of_the_annotation= Sentiment_df['User_of_the_annotation'][i])
        update_stmt_2 = update_stmt_1.where(The_Table_name.columns.Tweets_ID == int(Sentiment_df['Tweets_ID'][i][1:-1]))
        update_results2 = connection.execute(update_stmt_2)


    
    

    