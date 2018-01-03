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

class Main:
    '''Class para manipular la base de datos de Siata'''
    def __init__(self,codigo=None):
        self.codigo = codigo
        self.codigos = None
        self.colores_siata = [[0.69,0.87,0.93],[0.61,0.82,0.88],[0.32,0.71,0.77],[0.21,0.60,0.65],[0.0156,0.486,0.556],[0.007,0.32,0.36],[0.0078,0.227,0.26]]
    
    @property
    def nombre(self):
	'''Encuentra el nombre de la estacion'''
        return self.read_sqlOld("select NombreEstacion from estaciones where codigo = %s"%self.codigo)[0][0]        

    @property
    def longitud(self):
        return float(self.read_sqlOld("select Longitude from estaciones where codigo = %s"%self.codigo)[0][0]) 
    
    @property
    def latitud(self):
        return float(self.read_sqlOld("select Latitude from estaciones where codigo = %s"%self.codigo)[0][0]) 
    @property
    def municipio(self):
        '''Encuentra el municipio donde se encuentra ubicada la estacion'''
        return self.read_sqlOld("select Ciudad from estaciones where codigo = %s"%self.codigo)[0][0] 

    def mysql_settings(self,host="192.168.1.74",user="siata_Modif",passwd='M0d_si@t@64512',dbname="siata"):
        '''crea el entorno para entrar a la base de datos'''
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
    
    @staticmethod
    def read_csv(path,to_datetime = False,*args,**kwargs):
        '''Lee archivos planos .csv
        path = ruta donde se encuentra el archivo
        to_datetime para convertir a serie de tiempo'''
        df = pd.read_csv(rute,index_col=0,*args,**kwargs)
        if to_datetime:
            df.index = df.index.to_datetime()
        return df
    
    @staticmethod
    def filter_index(index):
        'Filtra indices malos en series de tiempo'
        good_date = []; bad_date = []
        for i in index:
            try:
                pd.to_datetime(i)
                good_date.append(i)
            except ValueError:
                bad_date.append(i)
                pass
        return good_date,bad_date
    
    @staticmethod
    def filter_quality(df,idcol='calidad',flag=1):
        'selects index by quality'
        good_quality,bad_quality = (df[df[idcol]==flag].index,df[df[idcol]<>flag].index)
        return good_quality,bad_quality

    @staticmethod
    def filter_negative(df):
        'selects index by positive and negative values'
        positive,negative = (df[df>0.0].index,df[df<0.0].index)
        return positive,negative
     
    @staticmethod
    def make_colormap(seq):
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
    @staticmethod
    def datetimeToString(DatetimeObject):
        '''convierte DatetimeObjet a %Y-%m-%d %H:%M'''
        return DatetimeObject.strftime('%Y-%m-%d %H:%M')    

    def read_sqlOld(self,query,toPandas=True):
        '''hace consultas en la base de datos
        query = sentencia para la base de datos
        ejm:
        query = 'describe estaciones'
        self.read_sqlOld(query),
        toPandas = True, convierte los datos en un dataFrame de pandas
        toPandas = False, devuelve los datos como una matriz'''
        self.mysql_settings() 
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        db_cursor = conn_db.cursor ()
        if self.codigos is None:
            db_cursor.execute (query)
        else:
            db_cursor.execute (query,self.codigos)
        if toPandas == True:
            data = pd.DataFrame(np.matrix(db_cursor.fetchall()))
        else:
            data = db_cursor.fetchall()
        conn_db.close()
        return data

    def read_sql(self,sql):
        '''hace consultas en la base de datos usando pandas
        output:
            DataFrame de pandas'''
        self.mysql_settings()
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        df = pd.read_sql(sql,conn_db)
        conn_db.close()
        return df
    
    @property
    def info(self):
        '''muestra información de la estacion
        output: pd.Series'''
        return self.infost.loc[self.codigo]
    
class Nivel(Main):
    '''Class Nivel
    Es una subclase de Main, que hereda sus propiedades'''    
    def __init__(self,codigo=None,codigos=None):
        Main.__init__(self,codigo)
        self.columns = ['id','NombreEstacion','Codigo','Red','estado','N','Ciudad','Longitude','Latitude','offsetN','action_level','minor_flooding','moderate_flooding','major_flooding']
        
    def __repr__(self):
        '''string to recreate the object'''
        return "Class Object. Nivel(codigo = {})".format(self.codigo)
    
    def __str__(self):
        '''string to recreate the main information of the object'''
        return 'Nombre: {}\nRed: Nivel\nCodigo: {}\nLongitud:{}\nLatitud: {}\nMunicipio: {}\nOffset: {}\nOffsetOld: {}\nriskLevels: {}'.format(self.nombre,self.codigo,self.longitud,self.latitud,self.municipio,self.offset,self.offsetOld,self.riskLevels)
    
    @property
    def infost(self):
        '''muestra informacion mas relevante de las estaciones de Nivel
        output: pd.DataFrame'''
        df = self.read_sql("SELECT %s FROM estaciones WHERE red = 'nivel' or red = 'nivel_mocoa'"%','.join(self.columns)).set_index('Codigo')
        df.index = np.array(df.index,int)
        return df
    
    @property
    def xSensor(self):
        '''Ubicacion del sensor en el eje x'''
        return pd.Series.from_array(dict(zip([128,108,245,109,106,186,124,135,247,140,96,101,246,92,94,245,238,251,1014,1013,260,158,182,93,239,90,104,143,183,240,99,91,115,116,134,152,166,179,155,236,173,178,196,195,259,268],[8.0,4.23,11.66,3.5,17.0,5.75,12.6,3.08,3.0,24.0,4.2,0.8,4.2,1.3,12.11,11.66,12.54,4.1,6.88,8.74,21.0,3.87,2.45,31.17,6.4,11.6,2.55,4.66,2.8,1.5,21.0,18.95,1.55,8.21,2.5,3.0,2.0,8.0,5.5,14.0,2.0,1.5,29.8,3.92,2.42,4.4])
    )).loc[self.codigo]
    
    @property
    def offsetOld(self):
        '''Encuentra el offset de las estaciones de nivel'''
        return float(self.read_sqlOld("select offsetN from estaciones where codigo = %s"%self.codigo)[0][0])
    
    @property
    def offset(self):
        '''Ultimo offset registrado'''
        try:
            offset = self.get_hbo.loc[self.codigo,'offset'].values[-1]
        except AttributeError:
            offset = self.get_hbo.loc[self.codigo,'offset']
        return offset
    @property
    def riskLevels(self):
        '''Niveles de riesgo
        output: tuple, size = 4'''
        return self.read_sqlOld("SELECT action_level, minor_flooding, moderate_flooding, major_flooding FROM estaciones WHERE codigo = '%s'"%self.codigo,toPandas=False)[0]
    
    @property
    def sensor_type(self):
        '''Encuentra el tipo de sensor, 
        ouput:
        1 si el sensor es de tipo radar,
        0 si el sensor es de tipo ultrasonido'''
        return int(self.read_sqlOld("select N from estaciones where codigo = %s"%self.codigo)[0][0])  
    
    @property
    def get_hbo(self):
        '''pd.DataFrame de historico_bancallena_offset
        hbo = historico_bancallena_offset'''
        df = self.read_sql('select codigo,fecha_hora,offset from historico_bancallena_offset')
        df[df==-9999]=np.NaN
        return df.set_index('codigo')
    
    
    def get_level_sensor(self,start,end):
        '''Obtiene datos del sensor a la lámina de agua
        input:
            (start,end) : (fecha inicial,fecha final), format = '%Y-%m-%d %H:%M,
        output:
            pd.Series con los datos del sensor, datos faltantes y con indices malos se llenan con NaN'''
        start = pd.to_datetime(start);end = pd.to_datetime(end)
        datef1 = "DATE_FORMAT(fecha,'%Y-%m-%d')";datef2 = "DATE_FORMAT(hora,'%H:%i:%s')" #formatos de fecha mysql
        # nueva forma de consultar
        query = "SELECT %s,%s,%s FROM datos WHERE cliente = '%s' and (((fecha>'%s') or (fecha = '%s' and hora>='%s')) and ((fecha<'%s') or (fecha = '%s' and hora<= '%s')))"%(datef1,datef2,['pr','NI'][self.sensor_type],self.codigo,start.strftime('%Y-%m-%d'),start.strftime('%Y-%m-%d'),start.strftime('%H:%M:00'),end.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),end.strftime('%H:%M:00'))
        df = self.read_sql(query)
        df.columns = ['fecha','hora','sensor']
        # serie de tiempo con los datos del sensor
        s = pd.Series(index=pd.to_datetime(self.datetimeToString(pd.date_range(start,end,freq='min'))))
        if df.empty:
            pass
        else:
            # si encuentra datos, concatena fecha,hora
            df.index = df.apply(lambda x:'%s %s'%(x[0],x[1][:-3]),axis=1).values
            # filtra indices datetime malos
            for fecha in df.index:
                try:
                    s[pd.to_datetime(fecha)] = df.loc[fecha,'sensor']
                except:
                    pass
        return s

    def offset_list(self,s):
        '''crea una serie de tiempo con el offset dinamico
        input:
            s = serie de tiempo con datos del sensor
        output:
            serie de tiempo con offset dinamico'''
        s = pd.Series.copy(s)
        for i in self.read_sqlOld("select fecha_hora,offset from historico_bancallena_offset where codigo = '%s';"%self.codigo,toPandas=False):
            s.loc[i[0]:] = i[1]
        return s
    
    
    def get_level(self,start,end,offset=None,bands=False):
        '''obtiene dataFrame con datos filtrados, el filtro se usa con la
        ventana movil de desviacion estandar de la media
        inputs:
            start,end = fecha inicial y final, format = %Y-%m-%d %H-%M'
            bands = True, muestra las bandas de confianza del filtro
        outputs:
            pd.DataFrame con datos filtrados'''
        # comienzo 19 minutos atras para aplicar ventana movil de 20 datos
        s = self.get_level_sensor(pd.to_datetime(start)-datetime.timedelta(minutes=19),end)
        # dataframe 
        df = pd.DataFrame(index = s.index)
        df['sensor'] = s
        # calculo de datos faltantes
        df['faltante'] = 0.0
        df.loc[df[s.isnull()].index,'faltante'] = 1
        # filtro desviacion estandar de la media
        stdOfMean = s.rolling(window=20,center=True).apply(lambda x: np.std(x)/np.sqrt(len(x)))
        s = s.iloc[18:]
        stdOfMean = stdOfMean.iloc[18:]
        upper = (s.rolling(window=2).mean()-5*stdOfMean).iloc[1:]
        lower = (s.rolling(window=2).mean()+5*stdOfMean).iloc[1:]
        filtro =(s.iloc[1:]<upper)|(s.iloc[1:]>lower)
        s = s.iloc[1:]
        if bands == True:
            df['upper'] = self.offset_list(df['sensor'])-upper
            df['lower'] = self.offset_list(df['sensor'])-lower 
        df = df.loc[start:end]
        df['filtrado'] = 0.0
        df.loc[df[filtro].index,'filtrado']=1.0
        # si es filtrado y faltante, prima faltante
        df.loc[df[df[['faltante','filtrado']].sum(axis=1)==2].index,'filtrado'] = 0.0
        if offset <> 'old':
            df['nivel'] = self.offset_list(df['sensor'])-df['sensor']
        else:
            df['nivel'] = self.offsetOld - df['sensor']
        # vuelve NaN datos filtrados
        df.loc[df[df[['faltante','filtrado']].sum(axis=1)<>0.0].index,'nivel']=np.NaN
        return df


