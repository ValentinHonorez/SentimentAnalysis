#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import select
from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData
import API_and_Database_function as func


connection, Twitter_Sentiment_Analysis = func.Database_Acces("mysql://root@localhost/sentimentanalysis", 'utf8' , 'Twitter_Sentiment_Analysis' )
 

#change the encoding in the Database_acces doesn't change anything for a select statement


# Build select statement
stmt = select([Twitter_Sentiment_Analysis.columns.Tweets_data]) #TODO : improving encoding 
#print(Twitter_Sentiment_Analysis.columns.CountryName)
#Twitter_Sentiment_Analysis.columns = [c.replace(' ', '_') for c in Twitter_Sentiment_Analysis.columns]
#stmt = stmt.where(Twitter_Sentiment_Analysis.columns.Language == 'fr')

# Execute the statement on connection and fetch 10 records: result
for results in connection.execute(stmt).fetchall() :
    
    print(results)



