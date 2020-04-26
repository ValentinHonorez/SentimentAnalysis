#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import select, update
from sqlalchemy import Table, Column, String, Integer, Float, Boolean, Date, BigInteger
from sqlalchemy import create_engine, MetaData
import API_and_Database_function as func
import pandas as pd
import re


connection, Twitter_Sentiment_Analysis = func.Database_Acces("mysql://root@localhost/sentiment?charset=utf8mb4", 'utf8' , 'Twitter_Sentiment_Analysis4' )
stmt = "SET NAMES 'UTF8';"
connection.execute(stmt)

#code si le tweet ID dans la base de données est un string


stmt = select([Twitter_Sentiment_Analysis.columns.Tweets_ID])  
#for results in connection.execute(stmt).fetchall(): 

final_Tweet_ID_db = []
for i in range (len(connection.execute(stmt).fetchall())) : 
    first = str(connection.execute(stmt).fetchall()[i])
    final = re.sub("',", "", first)
    final2 = re.sub("u'", "", final)
    final_Tweet_ID_db.append(final2)

#print(final_Tweet_ID_db[0:2])



Sentiment_df = pd.read_csv("Export_csv5.csv", encoding = 'UTF-16 LE',  sep = '\t')
#print(Sentiment_df['Tweets_ID'])


    

for i in range(len(Sentiment_df['Tweets_ID'])) : #range du excel
    for j in range (len(final_Tweet_ID_db)): # range de la database
        if final_Tweet_ID_db[i] == Sentiment_df['Tweets_ID'][j] :
            #print('updating')
                
            update_stmt = update(Twitter_Sentiment_Analysis).values(Manual_Sentiment_Annotation= Sentiment_df['Manual_Sentiment_Annotation'][j]) #alors on update l'annotation dans la database à partir du excel (donc j)
            update_stmt2 = update_stmt.where(Twitter_Sentiment_Analysis.columns.Tweets_ID == connection.execute(stmt).fetchall()[i]) #cette update se passe dans la bonne colonne de la database (donc i)
            update_results = connection.execute(update_stmt2)
        else : 
            #print('not updating')
            j = j+1
       





#to see if it worked
# select_stmt = select([Twitter_Sentiment_Analysis]).where(Twitter_Sentiment_Analysis.columns.Tweets_ID == final_Tweet_ID_db[0])
# new_results = connection.execute(select_stmt).fetchall()
# print(new_results)

