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
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.pyplot as plt
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
        self.radar_path = '/media/nicolas/Home/nicolas/101_RadarClass/'
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

    	self.colores_siata = [[0.69,0.87,0.93],[0.61,0.82,0.88],[0.32,0.71,0.77],[0.21,0.60,0.65],\
                          [0.0156,0.486,0.556],[0.007,0.32,0.36],[0.0078,0.227,0.26]]

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
        print query
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
        if files:
            for file in files:
                comienza,finaliza,codigo,usuario = self.file_format_to_variables(file)
                if (comienza<=start) and (finaliza>=end) and (codigo==self.codigo):
                    file =  file[:file.find('.')]
                    break
                else:
                    file = None
        else:
            file = None
        return file
	
    def radar_rain(self,start,end,ext='.hdr'):
        '''
        Reads rain fields (.bin or .hdr)
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame or Series with mean radar rain
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        file = self.check_rain_files(start,end)
        if file:
            file = self.rain_path+file
            if ext == '.hdr':
                obj =  self.hdr_to_series(file+'.hdr')
            else:
                print file
                obj =  self.bin_to_df(file)
            obj = obj.loc[start:end]
        else:
            print 'converting rain data, it may take a while'
            converter = os.path.dirname(os.getcwd())+'/cprv1/RadarConvStra2Basin2.py'
            save =  '%s%s'%(self.rain_path,self.file_format(start,end))
            self.get_radar_rain(start,end,self.info.nc_path,self.radar_path,save,converter=converter,utc=True)
            print file
            file = self.rain_path + self.check_rain_files(start,end)
            if ext == '.hdr':
                obj =  self.hdr_to_series(file+'.hdr')
            else:
                obj =  self.bin_to_df(file)
            obj = obj.loc[start:end]
        return obj

    def radar_rain_vect(self,start,end):
        '''
        Reads rain fields (.bin)
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with datetime index and basin radar fields
        '''
        return self.radar_rain(start,end,ext='.bin')

    def level(self,start,end,offset='new'):
        '''
        Reads rain fields (.bin)
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with datetime index and basin radar fields
        '''
        sql = SqlDb(codigo = self.codigo,**self.remote_server)
        s = sql.fecha_hora_format_data(['pr','NI'][self.info.tipo_sensor],start,end)
        if offset == 'new':
            return self.info.offset - s
        else:
            return self.info.offset_old - s

    def offset_remote(self):
        remote = SqlDb(**self.remote_server)
        query = "SELECT codigo,fecha_hora,offset FROM historico_bancallena_offset"
        df = remote.read_sql(query).set_index('codigo')
        try:
            offset = float(df.loc[self.codigo,'offset'])
        except TypeError:
            offset =  df.loc[self.codigo,['fecha_hora','offset']].set_index('fecha_hora').sort_index()['offset'][-1]
        return offset


    def mysql_query(self,query,toPandas=True):
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        db_cursor = conn_db.cursor ()
        db_cursor.execute (query)
        if toPandas == True:
            data = pd.DataFrame(np.matrix(db_cursor.fetchall()))
        else:
            data = db_cursor.fetchall()
        conn_db.close()
        return data

    def last_bat(self):
        dfl = self.mysql_query('select * from levantamiento_aforo_nueva')
        dfl.columns = self.mysql_query('describe levantamiento_aforo_nueva')[0].values
        dfl = dfl.set_index('id_aforo')
        for id_aforo in list(set(dfl.index)):
            id_estacion_asociada,fecha = self.mysql_query("SELECT id_estacion_asociada,fecha from aforo_nueva where id_aforo = %s"%id_aforo,toPandas=False)[0]
            dfl.loc[id_aforo,'id_estacion_asociada'] = int(id_estacion_asociada)
            dfl.loc[id_aforo,'fecha'] = fecha
        dfl = dfl.reset_index().set_index('id_estacion_asociada')
        lev = dfl[dfl['fecha']==max(list(set(pd.to_datetime(dfl.loc[self.codigo,'fecha'].values))))][['x','y']].astype('float')
        cond = (lev['x']<self.info.x_sensor).values
        flag = cond[0]
        for i,j in enumerate(cond):
            if j==flag:
                pass
            else:
                point = (tuple(lev.iloc[i-1].values),tuple(lev.iloc[i].values))
            flag = j
        intersection = self.line_intersection(point,((self.info.x_sensor,0.1*lev['y'].min()),(self.info.x_sensor,1.1*lev['y'].max(),(self.info.x_sensor,))))
        lev = lev.append(pd.DataFrame(np.matrix(intersection),index=['x_sensor'],columns=['x','y'])).sort_values('x')
        lev['y'] = lev['y']-intersection[1]
        lev.index = range(1,lev.index.size+1)
        return lev

    def get_sections(self,levantamiento,level):
            hline = ((levantamiento['x'].min()*1.1,level),(levantamiento['x'].max()*1.1,level)) # horizontal line
            lev = pd.DataFrame.copy(levantamiento) #df to modify
            #PROBLEMAS EN LOS BORDES
            borderWarning = 'Warning:\nProblemas de borde en el levantamiento'
            if lev.iloc[0]['y']<level:
                print '%s en banca izquierda'%borderWarning
                lev = pd.DataFrame(np.matrix([lev.iloc[0]['x'],level]),columns=['x','y']).append(lev)
            if lev.iloc[-1]['y']<level:
                print '%s en banca derecha'%borderWarning
                lev = lev.append(pd.DataFrame(np.matrix([lev.iloc[-1]['x'],level]),columns=['x','y']))
            condition = (lev['y']>=level).values
            flag = condition[0]
            nlev = []
            intCount = 0
            ids=[]
            for i,j in enumerate(condition):
                if j==flag:
                    ids.append(i)
                    nlev.append(list(lev.iloc[i].values))
                else:
                    intCount+=1
                    ids.append('Point %s'%intCount)
                    line = (list(lev.iloc[i-1].values),list(lev.iloc[i].values)) #  #puntoA
                    inter = self.line_intersection(line,hline)
                    nlev.append(inter)
                    ids.append(i)
                    nlev.append(list(lev.iloc[i].values))
                flag = j
            df = pd.DataFrame(np.matrix(nlev),columns=['x','y'],index=ids)
            dfs = []
            for i in np.arange(1,100,2)[:intCount/2]:
                dfs.append(df.loc['Point %s'%i:'Point %s'%(i+1)])
            return dfs

    @staticmethod
    def get_area(x,y):
        '''Calcula las áreas y los caudales de cada
        una de las verticales, con el método de mid-section
        Input:
        x = Distancia desde la banca izquierda, type = numpy array
        y = Produndidad
        Output:
        area = Área de la subsección
        Q = Caudal de la subsección
        '''
        # cálculo de áreas
        d = np.absolute(np.diff(x))/2.
        b = x[:-1]+ d
        area = np.diff(b)*y[1:-1]
        area = np.insert(area, 0, d[0]*y[0])
        area = np.append(area,d[-1]*y[-1])
        area = np.absolute(area)
        # cálculo de caudal
        return area

    def get_areas(self,dfs):
        area = 0
        for df in dfs:
            area+=sum(self.get_area(df['x'].values,df['y'].values))
        return area

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)

    def longitude_latitude_basin(self):
        mcols,mrows = wmf.cu.basin_2map_find(self.structure,self.ncells)
        mapa,mxll,myll=wmf.cu.basin_2map(self.structure,self.structure[0],mcols,mrows,self.ncells)
        longs = np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mcols)])
        lats  = np.array([myll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mrows)])
        return longs,lats

    def basin_mappable(self,vec=None, extra_long=0,extra_lat=0,perimeter_keys={},contour_keys={},**kwargs):
        longs,lats=self.longitude_latitude_basin()
        x,y=np.meshgrid(longs,lats)
        y=y[::-1]
        # map settings
        m = Basemap(projection='merc',llcrnrlat=lats.min()-extra_lat, urcrnrlat=lats.max()+extra_lat,
            llcrnrlon=longs.min()-extra_long, urcrnrlon=longs.max()+extra_long, resolution='c',**kwargs)
        # perimeter plot
        xp,yp = m(self.Polygon[0], self.Polygon[1])
        m.plot(xp, yp,**perimeter_keys)
        # vector plot
        if vec is not None:
            map_vec,mxll,myll=wmf.cu.basin_2map(self.structure,vec,len(longs),len(lats),self.ncells)
            map_vec[map_vec==wmf.cu.nodata]=np.nan
            xm,ym=m(x,y)
            contour = m.contourf(xm, ym, map_vec.T, 25,**contour_keys)
        else:
            contour = None
        return m,contour

    def adjust_basin(self,rel=0.766,fac=0.0):
        longs,lats = self.longitude_latitude_basin()
        x = longs[-1]-longs[0]
        y = lats[-1] - lats[0]
        if x>y:
            extra_long = 0
            extra_lat = (rel*x-y)/2.0
        else:
            extra_lat=0
            extra_long = (y/(2.0*rel))-(x/2.0)
        return extra_lat+fac,extra_long+fac


    def radar_cmap(self):
        bar_colors=[(255, 255, 255),(0, 255, 255), (0, 0, 255),(70, 220, 45),(44, 141, 29),\
                       (255,255,75),(255,142,0),(255,0,0),(128,0,128),(102,0,102),(255, 153, 255)]
        lev = np.array([0.,1.,5.,10.,20.,30.,45.,60., 80., 100., 150.])
        scale_factor =  ((255-0.)/(lev.max() - lev.min()))
        new_Limits = list(np.array(np.round((lev-lev.min())*\
                                    scale_factor/255.,3),dtype = float))
        Custom_Color = map(lambda x: tuple(ti/255. for ti in x) , bar_colors)
        nueva_tupla = [((new_Limits[i]),Custom_Color[i],) for i in range(len(Custom_Color))]
        cmap_radar =colors.LinearSegmentedColormap.from_list('RADAR',nueva_tupla)
        levels_nuevos = np.linspace(np.min(lev),np.max(lev),255)
        norm_new_radar = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=256)
        return cmap_radar,levels_nuevos,norm_new_radar

    def level_local(self,start,end,offset='new'):
        if offset=='new':
            offset = self.info.offset
        else:
            offset = self.info.offset_old
        format = (self.codigo,start,end)
        query = "select fecha,nivel from hydro where codigo='%s' and fecha between '%s' and '%s';"%format
        return (offset - self.read_sql(query).set_index('fecha')['nivel'])

    def convert_level_to_risk(self,value,risk_levels):
        ''' Convierte lamina de agua o profundidad a nivel de riesgo
        Parameters
        ----------
        value : float. Valor de profundidad o lamina de agua
        riskLevels: list,tuple. Niveles de riesgo

        Returns
        -------
        riskLevel : float. Nivel de riesgo
        '''
        if math.isnan(value):
            return np.NaN
        else:
            dif = value - np.array([0]+list(risk_levels))
            return int(np.argmin(dif[dif >= 0]))

    @property
    def risk_levels(self):
        query = "select n1,n2,n3,n4 from estaciones_estaciones where codigo = '%s'"%self.codigo
        return tuple(self.read_sql(query).values[0])

    def risk_level_series(self,start,end):
        return self.level_local(start,end).apply(lambda x: self.convert_level_to_risk(x,self.risk_levels))
    
    def risk_level_df(self,start,end):
        print 'Making risk dataframe'
        df = pd.DataFrame(index=pd.date_range(start,end,freq='D'),columns=self.infost.index)
        for count,codigo in enumerate(df.columns):
            print "%s | '%.2f %%' - %s out of %s "%(codigo,(count+1)*100.0/df.columns.size,count+1,df.columns.size)
            try:
                clase = Nivel(user=self.user,codigo=codigo,passwd=self.passwd,**info.LOCAL)
                df[codigo] = clase.risk_level_series(start,end).resample('D',how='max')
            except:
                df[codigo] = np.NaN
                print "WARNING: station %s empty,row filled with NaN"%codigo
        print 'risk dataframe finished'
        return df

    def plot_level(self,level,rain,riesgos,fontsize=14,ncol=4,ax=None,bbox_to_anchor=(1.0,1.2),**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(13.,4))
            ax = fig.add_subplot(111)
        nivel = level.resample('H',how='mean')
        nivel.plot(ax=ax,label='',color='k')
        nivel.plot(alpha=0.3,label='',color='r',fontsize=fontsize,**kwargs)
        axu= ax.twinx()
        axu.set_ylabel('Lluvia promedio [mm/h]',fontsize=fontsize)
        mean_rain = (rain*60/5.0).resample('H',how='sum')

        mean_rain.plot(ax=axu,alpha=0.5,fontsize=fontsize,**kwargs)
        axu.fill_between(mean_rain.index,0,mean_rain.values,alpha=0.2)
        ylim = axu.get_ylim()[::-1]
        ylim = (ylim[0],0.0)
        axu.set_ylim(ylim)
        ax.set_ylabel('Nivel (cm)',fontsize=fontsize)
        alpha=0.2
        ax.fill_between(nivel.index,ax.get_ylim()[0],riesgos[0],alpha=0.1,color=self.colores_siata[0])
        ax.fill_between(nivel.index,riesgos[0],riesgos[1],alpha=alpha,color='green')
        ax.fill_between(nivel.index,riesgos[1],riesgos[2],alpha=alpha,color='orange')
        ax.fill_between(nivel.index,riesgos[2],riesgos[3],alpha=alpha,color='red')
        ax.fill_between(nivel.index,riesgos[3],ax.get_ylim()[1],alpha=alpha,color='indigo')
        ax.set_ylim(0,max(riesgos)*1.05)
        return (ax,axu)
