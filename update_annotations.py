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
func.update_annotations_db(Twitter_Sentiment_Analysis, connection, "Export_csv5.csv")