#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData




engine = create_engine("mysql://root@localhost/sentimentanalysis", echo = True, encoding = 'mbcs')

metadata = MetaData()

connection = engine.connect()

#create_str = "CREATE DATABASE IF NOT EXISTS sentimentanalysis ;"
#engine.execute(create_str)




Twitter_Sentiment_Analysis = Table('Twitter_Sentiment_Analysis', metadata,
             Column('Database_ID', BigInteger), 
             Column('Tweets_ID', BigInteger),
             Column('Tweets_data', String(512)),
             Column('Tweets_Date_of_creation', Date),
             Column('Retweet', Boolean),
             Column('Retweet_original_TweetID', BigInteger),
             Column('Retweet_original_authorID', BigInteger),
             Column('Retweet_original_author_Name', String(256)),
             Column('Retweet_original_author_Screen_Name', String(256)),
             


             Column( 'Reply', Boolean),
             Column('Reply_original_authorID', BigInteger),
             Column('Reply_original_author_Screen_Name', String(256)),



            
             Column('User_ID', BigInteger),
             Column( 'USERNAME', String(256)),
             Column('SCREEN_NAME', String(256)),
             Column('City_Name', String(256)),
             Column( 'Country_Name', String(256)),
             Column('Language', String(256)),
             Column( 'Verified_account', Boolean),
             Column('Numbers_of_followers', Integer),
             Column( 'Numbers_of_friends', Integer),
             Column('User_mentions', String(256)),





             Column('Hashtags', Boolean),
             Column( 'Other_symbols', Boolean),
             Column('Video', String(256)),
             Column('Image', String(256)),
             Column('URLS', String(256)),

             Column('Manual_Sentiment_Annotation', Integer),
             Column('User_of_the_annotation', String(256)))
                         


# Use the metadata to create the table
metadata.create_all(engine)

