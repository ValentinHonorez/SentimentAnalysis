#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import API_and_Database_function as func


connection, Twitter_Sentiment_Analysis = func.Database_Acces("mysql://root@localhost/sentiment?charset=utf8mb4", 'utf8' , 'Twitter_Sentiment_Analysis4' )
stmt = "SET NAMES 'UTF8';"
connection.execute(stmt)
func.export_to_csv(Twitter_Sentiment_Analysis, connection, "Export_csv2.csv")