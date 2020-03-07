
from sqlalchemy import delete
from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData

from sqlalchemy import insert, select
import API_and_Database_function as func


connection, Twitter_Sentiment_Analysis = func.Database_Acces("mysql://root@localhost/sentimentanalysis", 'utf8' , 'Twitter_Sentiment_Analysis' )
func.All_tweets_Delete(Twitter_Sentiment_Analysis, connection)
