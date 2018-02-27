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
import information as info
from wmf import wmf
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
        self.execute_sql(execution,*keys,**kwargs)

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
        format = (how,date_field_name,self.table,name,codigo)
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
    # functions for id_hydro
    @property
    def info(self):
        '''
        Gets full information from single station
        ---------
        pd.Series
        '''
        query = "SELECT * FROM %s"%self.table
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

    @staticmethod
    def fecha_hora_query(start,end):
        '''
        Efficient way to query in tables with fields fecha,hora
        such as table datos
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        Alternative query between two datetime objects
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        def f(date):
            return tuple([date.strftime('%Y-%m-%d')]*2+[date.strftime('%H:%M:00')])
        query = "("+\
                "((fecha>'%s') or (fecha='%s' and hora>='%s'))"%f(start)+" and "+\
                "((fecha<'%s') or (fecha='%s' and hora<='%s'))"%f(end)+\
                ")"
        return query

    def fecha_hora_format_data(self,field,start,end):
        '''
        Gets pandas Series with data from tables with
        date format fecha and hora detached, and filter
        bad data
        Parameters
        ----------
        field        : Sql table field name
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas time Series
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        format = (field,self.codigo,self.fecha_hora_query(start,end))
        df = self.read_sql("SELECT fecha,hora,%s from datos WHERE cliente = '%s' and calidad = '1' and %s"%format)
        # converts centiseconds in 0
        df['hora'] = df['hora'].apply(lambda x:x[:-3]+':00')
        # concatenate fecha and hora fields, and makes nan bad datetime indexes
        df.index= pd.to_datetime(df['fecha'] + ' '+ df['hora'],errors='coerce')
        df = df.sort_index()
        # removes nan
        df = df.loc[df.index.dropna()]
        # masks duplicated index
        df[df.index.duplicated(keep=False)]=np.NaN
        df = df.dropna()
        # drops coluns fecha and hora
        df = df.drop(['fecha','hora'],axis=1)
        # reindex to have all indexes in full time series
        new_index = pd.date_range(start,end,freq='min')
        series = df.reindex(new_index)[field]
        return series


class Nivel(SqlDb,wmf.SimuBasin):
    '''
    Provide functions to manipulate data related
    to a level sensor and its basin.
    '''
    local_table  = 'estaciones_estaciones'
    remote_table = 'estaciones'
    def __init__(self,user,passwd,codigo = None,SimuBasin = False,remote_server = info.REMOTE,**kwargs):
        '''
        The instance inherits modules to manipulate SQL
        data and uses (hidrology modeling framework) wmf
        Parameters
        ----------
        codigo        : primary key
        remote_server :
        local_server  : database kwargs to pass into the Sqldb class
        nc_path       : path of the .nc file to set wmf class
        '''
        self.remote_server = remote_server
        self.data_path ='/media/nicolas/maso/Mario/'
        self.rain_path = self.data_path + 'user_output/radar/'
        self.radar_path = self.data_path + 'radar'
        if not kwargs:
            kwargs = info.LOCAL
        SqlDb.__init__(self,codigo=codigo,user=user,passwd=passwd,**kwargs)
        if SimuBasin:
            query = "SELECT nc_path FROM %s WHERE codigo = '%s'"%(self.local_table,self.codigo)
            try:
                nc_path = self.read_sql(query)['nc_path'][0]
            except:
                nc_path = self.data_path + 'basins/%s.nc'%self.codigo
            wmf.SimuBasin.__init__(self,rute=nc_path)

    @property
    def info(self):
        query = "SELECT * FROM %s WHERE clase = 'Nivel' and codigo='%s'"%(self.local_table,self.codigo)
        s = self.read_sql(query).T
        return s[s.columns[0]]

    @property
    def infost(self):
        '''
        Gets full information from all stations
        Returns
        ---------
        pd.DataFrame
        '''
        query = "SELECT * FROM %s WHERE clase ='Nivel'"%(self.local_table)
        return self.read_sql(query).set_index('codigo')

    @staticmethod
    def get_radar_rain(start,end,nc_path,radar_path,save,
                    converter = 'RadarConvStra2Basin2.py',
                    utc=False,
                    dt = 300,*keys,**kwargs):
        '''
        Convert radar rain to basin
        Parameters
        ----------
        start         : inicial date
        end           : final date
        nc_path       : path to nc basin file
        radar_path    : path to radar data
        save          : path to save
        converter     : path of main rain converter script,
                        default RadarConvStra2Basin2.py
        utc           : if radar data is in utc
        dt            : timedelta, default = 5 minutes
        Returns
        ----------
        bin, hdr files with rain data
        '''
        start = pd.to_datetime(start); end = pd.to_datetime(end)
        if utc ==True:
            delay = datetime.timedelta(hours=5)
            start = start+delay
            end = end + delay
        hora_inicial = start.strftime('%H:%M')
        hora_final = end.strftime('%H:%M')
        format = (
                converter,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                nc_path,
                radar_path,
                save,
                dt,
                hora_inicial,
                hora_final
                 )
        query = '%s %s %s %s %s %s -t %s -v -s -1 %s -2 %s'%format
        output = os.system(query)
        if output != 0:
            print 'ERROR: something went wrong'
        return query

    @staticmethod
    def hdr_to_series(path):
        '''
        Reads hdr rain files and converts it into pandas Series
        Parameters
        ----------
        path         : path to .hdr file
        Returns
        ----------
        pandas time Series with mean radar rain
        '''
        s =  pd.read_csv(path,skiprows=5,usecols=[2,3]).set_index(' Fecha ')[' Lluvia']
        s.index = pd.to_datetime(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],s.index))
        return s

    @staticmethod
    def hdr_to_df(path):
        '''
        Reads hdr rain files and converts it into pandas DataFrame
        Parameters
        ----------
        path         : path to .hdr file
        Returns
        ----------
        pandas DataFrame with mean radar rain
        '''
        if path.endswith('.hdr') != True:
            path = path+'.hdr'
        df = pd.read_csv(path,skiprows=5).set_index(' Fecha ')
        df.index = pd.to_datetime(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],df.index))
        df = df.drop('IDfecha',axis=1)
        df.columns = ['record','mean_rain']
        return df

    def bin_to_df(self,path,start=None,end=None,**kwargs):
        '''
        Reads rain fields (.bin) and converts it into pandas DataFrame
        Parameters
        ----------
        path         : path to .hdr and .bin file
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with mean radar rain
        Note
        ----------
        path without extension, ejm folder_path/file not folder_path/file.bin,
        if start and end is None, the program process all the data
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        df = self.hdr_to_df(path)
        if (start is not None) and (end is not None):
            df = df.loc[start:end]
        df = df[df['record']!=1]
        records = df['record'].values
        rain_field = []
        for count,record in enumerate(records):
            rain_field.append(wmf.models.read_int_basin('%s.bin'%path,record,self.ncells)[0])
            count = count+1
            format = (count*100.0/len(records),count,len(records))
            print("progress: %.1f %% - %s out of %s"%format)
        return pd.DataFrame(np.matrix(rain_field),index=df.index)

    def file_format(self,start,end):
        '''
        Returns the file format customized for siata for elements containing
        starting and ending point
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        file format with datetimes like %Y%m%d%H%M
        Example
        ----------
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        format = '%Y%m%d%H%M'
        return '%s-%s-%s-%s'%(start.strftime(format),end.strftime(format),self.codigo,self.user)

    def file_format_date_to_datetime(self,string):
        '''
        Transforms string in file_format like string to datetime object
        Parameters
        ----------
        string         : string object in file_format like time object
        Returns
        ----------
        datetime object
        Example
        ----------
        In : self.file_format_date_to_datetime('201707141212')
        Out: Timestamp('2017-07-14 12:12:00')
        '''
        format = (string[:4],string[4:6],string[6:8],string[8:10],string[10:12])
        return pd.to_datetime("%s-%s-%s %s:%s"%format)

    def file_format_to_variables(self,string):
        '''
        Splits file name string in user and datetime objects
        Parameters
        ----------
        string         : file name
        Returns
        ----------
        (user,start,end) - (string,datetime object,datetime object)
        '''
        string = string[:string.find('.')]
        start,end,codigo,user = list(x.strip() for x in string.split('-'))
        start,end = self.file_format_date_to_datetime(start),self.file_format_date_to_datetime(end)
        return start,end,int(codigo),user

    def check_rain_files(self,start,end):
        '''
        Finds out if rain data has already been processed
        start        : initial date
        end          : final date
        Returns
        ----------
        file path or None for no coincidences
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        files = os.listdir(self.rain_path)
        for file in files:
            comienza,finaliza,codigo,usuario = self.file_format_to_variables(file)
            if (comienza<=start) and (finaliza>=end) and (codigo==self.codigo):
                file =  file[:file.find('.')]
                break
            else:
                file =  None
            return file

    def mean_rain_vect(self,path):
        pass

    def mean_rain(self,start,end):
        pass

    def sensor(start,end,remote=False):
        pass

    def surface_velocity(start,end,remote=False):
        pass

    def camera(start,end):
        pass

    def pluvios_in_basin(start,end):
        pass

    def offset_remote(self):
        remote = SqlDb(**self.remote_server)
        query = "SELECT codigo,fecha_hora,offset FROM historico_bancallena_offset"
        df = remote.read_sql(query).set_index('codigo')
        try:
            offset = float(df.loc[self.codigo,'offset'])
        except TypeError:
            offset =  df.loc[self.codigo,['fecha_hora','offset']].set_index('fecha_hora').sort_index()['offset'][-1]
        return offset
