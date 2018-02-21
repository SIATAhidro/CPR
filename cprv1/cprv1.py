#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

import MySQLdb
import pandas as pd
import numpy as np
import datetime
import math
import time
import mysql.connector
from sqlalchemy import create_engine
import os
import warnings
import static as st
import bookplots as bp

warnings.filterwarnings('ignore')

class SqlDb:
    '''
    Class para manipular las bases de datos SQL
    '''
    str_date_format = '%Y-%m-%d %H:%M:00'

    def __init__(self,dbname,user,host,passwd,port,table=None,codigo=None,
                *keys,**kwargs):
        '''
        instance and properties
        '''
        self.table  = table
        self.host   = host
        self.user   = user
        self.passwd = passwd
        self.dbname = dbname
        self.port   = port
        self.codigo = codigo

    def __repr__(self):
        '''string to recreate the object'''
        return "codigo = {}".format(self.codigo)

    def __str__(self):
        '''string to recreate the main information of the object'''
        return 'dbname: {}, user: {}'.format(self.dbname,self.user)

    @property
    def conn_db(self):
        '''
        Engine connection: makes possible connection with SQL database
        '''
        conn_db = MySQLdb.connect(self.host,self.user,self.passwd,self.dbname)
        return conn_db

    def logger(self,function,status,message):
        '''
        Logs methods performance
        Returns
        -------
        string comma separated values'''
        now = datetime.datetime.now().strftime(self.str_date_format)
        return '%s,%s,%s,%s'%(now,self.user,function,status,message)

    def read_sql(self,sql,close_db=True,*keys,**kwargs):
        '''
        Read SQL query or database table into a DataFrame.
        Parameters
        ----------
        sql : string SQL query or SQLAlchemy Selectable (select or text object)
            to be executed, or database table name.

        keys and kwargs = ( sql, con, index_col=None, coerce_float=True,
                            params=None, parse_dates=None,columns=None,
                            chunksize=None)
        Returns
        -------
        DataFrame
        '''
        df = pd.read_sql(sql,self.conn_db,*keys,**kwargs)
        if close_db == True:
            self.conn_db.close()
        return df

    def execute_sql(self,query,close_db=True):
        '''
        Execute SQL query or database table into a DataFrame.
        Parameters
        ----------
        query : string SQL query or SQLAlchemy Selectable (select or text object)
            to be executed, or database table name.
        keys = (sql, con, index_col=None, coerce_float=True, params=None,
        parse_dates=None,
        columns=None, chunksize=None)
        Returns
        -------
        DataFrame'''
        conn_db = self.conn_db
        conn_db.cursor().execute(query)
        conn_db.commit()
        if close_db == True:
            conn_db.close ()
        #print (self.logger('execute_mysql','execution faile','worked',query))

    def insert_data(self,fields,values,*keys,**kwargs):
        '''
        inserts data into SQL table from list of fields and values
        Parameters
        ----------
        fields   = list of fields names from SQL db
        values   = list of values to be inserted
        Example
        -------
        insert_data(['fecha','nivel'],['2017-07-13',0.5])
        '''
        values = str(values).strip('[]')
        fields = str(fields).strip('[]').replace("'","")
        execution = 'INSERT INTO %s (%s) VALUES (%s)'%(self.table,fields,values)
        self.execute_sql(query,*keys,**kwargs)

    def update_data(self,field,value,pk,*keys,**kwargs):
        '''
        Update data into SQL table
        Parameters
        ----------
        fields   = list of fields names from SQL db
        values   = list of values to be inserted
        pk       = primary key from table
        Example
        -------
        update_data(['nivel','prm'],[0.5,0.2],1025)
        '''
        query = "UPDATE %s SET %s = '%s' WHERE id = '%s'"%(self.table,field,value,pk)
        self.execute_sql(query,*keys,**kwargs)

    def read_boundary_date(self,how,date_field_name = 'fecha'):
        '''
        Gets boundary date from SQL table based on DateField or DatetimeField name
        Parameters
        ----------
        how             = method to get boundary, could be max or min
        date_field_name = field name in Table
        Example
        -------
        read_bound_date('min')
        '''
        format = (how,date_field,self.table,name,codigo)
        return self.read_sql("select %s(%s) from %s where codigo='%s'"%format).loc[0,'%s(fecha)'%how]

    def df_to_sql(self,df,chunksize=20000,*keys,**kwargs):
        '''Replaces existing table with dataframe
        Parameters
        ----------
        df        = Pandas DataFrame to replace table
        chunksize = If not None, then rows will be written in batches
        of this size at a time
        '''
        format = (self.user,self.passwd,self.host,self.port,)
        engine = create_engine('mysql+mysqlconnector://%s:%s@%s:%s/cpr'%format,echo=False)
        df.to_sql(name      = self.table,
                  con       = engine,
                  if_exists = 'replace',
                  chunksize = chunksize,
                  index     = False,
                  *keys,**kwargs)

    def bound_date(self,how,date_field_name='fecha'):
        '''
        Gets firs and last dates from date field name of SQL table
        Parameters
        ----------
        how                = min or max (ChoiseField),
        date_field_name    = field name of SQL table, containing datetime,
        timestamp or other time formats
        Returns
        ----------
        DateTime object
        '''
        format = (how,date_field_name,self.table,self.codigo)
        return self.read_sql("select %s(%s) from %s where codigo='%s'"%format).loc[0,'%s(%s)'%(how,date_field_name)]

        Returns
    # functions for id_hydro
    @property
    def info(self):
        '''
        Gets full information from single station
        ---------
        pd.Series
        '''
        query = "SELECT * FROM %s WHERE codigo='%s'"%(self.table,self.codigo)
        return self.read_sql(query).T[0]

    def update_series(self,series,field):
        '''
        Update table from pandas time Series
        Parameters
        ----------
        series   = pandas time series with datetime or timestamp index
        and frequency = '5min'
        field    = field to be update
        Example
        value = series[fecha]
        ----------
        series = pd.Series(...,index=pd.date_range(...))
        update_series(series,'nivel')
        this updates the field nivel
        '''
        pk = self.id_df
        for fecha in series.index:
            if math.isnan(value):
                pass
            else:
                id    = pk[fecha]
                self.update_data(field,value,id)
