#!/usr/bin/env python
# -*- coding: utf-8 -*-
from wmf import wmf
import numpy as np
import pylab as pl
import pandas as pd
import pickle
import time
import datetime
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import collections
import warnings
warnings.filterwarnings('ignore')
import scipy as scp
import os
import glob
import aforos as af
from IPython.display import IFrame
import matplotlib as mpl
import matplotlib.colors as mcolors
import xlsxwriter
import codecs
from multiprocessing import Pool
from IPython.core.display import HTML
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import MySQLdb
import matplotlib.font_manager as fm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib import colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import gdal
import locale
import math
import sys
from scipy import spatial
from StringIO import StringIO
from reportlab.pdfgen import canvas
from PDFImageSIATA import PdfImageSIATA
from pyPdf import PdfFileWriter, PdfFileReader
from matplotlib.patches import Rectangle
import matplotlib.dates as dates
import matplotlib.dates as mdates

#plt.rc('text', usetex=True)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# SETTING PLOTS ENVIRONMENT
plt.style.use('seaborn-dark')
plt.rc('font', family=fm.FontProperties(fname='/media/nicolas/Home/Jupyter/MarioLoco/Tools/AvenirLTStd-Book.ttf',).get_name())
typColor = '#%02x%02x%02x' % (8,31,45)
plt.rc('axes',labelcolor=typColor)
plt.rc('axes',edgecolor=typColor)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)


class SiataData:
    '''Class hecha para manipular la base de datos de Siata'''
    def __init__(self,codigo=None,codigos=None):
        self.codigo = codigo
        self.codigos = codigos
        self.colores_siata = [[0.69,0.87,0.93],[0.61,0.82,0.88],[0.32,0.71,0.77],[0.21,0.60,0.65],\
                              [0.0156,0.486,0.556],[0.007,0.32,0.36],[0.0078,0.227,0.26]]

    def mysql_settings(self,host="192.168.1.74",user=None,passwd=None,dbname="siata"):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname

    def mysql_desc_table(self,table):
        return list(pd.DataFrame(np.matrix(self.mysql_query("describe %s;"%table))).set_index(0).index)

    @staticmethod
    def read_csv(rute,datetimeIndex = False,*args,**kwargs):
        df = pd.read_csv(rute,index_col=0,*args,**kwargs)
        if datetimeIndex:
            df.index = df.index.to_datetime()
        return df

    @staticmethod
    def roundTime(time,each=5.0):
        redondeo = int(math.ceil(time.minute / each)) * each
        return time + datetime.timedelta(minutes = redondeo-time.minute)


    @staticmethod
    def filter_index(index):
        'selects index by good datetime convertion'
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
    def curva_duracion(df):
        a,b = np.histogram(df,bins=50)
        x = 100-np.cumsum(a/np.sum(np.array(a,dtype=float))*100.)
        y = (b[1:] + b[:-1] )/2.
        x[0] = 99.9
        return x,y

    @staticmethod
    def filter_negative(df):
        'selects index by positive and negative values'
        positive,negative = (df[df>0.0].index,df[df<0.0].index)
        return positive,negative

    @staticmethod
    def filter_percentile(s,quantile=0.9998):
        s = s.dropna()
        return s[s>s.quantile(quantile)].index

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



    def Read_DEM(self,*args,**kwargs):
        self.DataSet = gdal.Open(kwargs.get('path','../Tools/DemAM.tif'))
        self.GeoTransform =  self.DataSet.GetGeoTransform()
        self.DEM =  pd.DataFrame(self.DataSet.ReadAsArray(),\
                                 columns=np.array([self.GeoTransform[0]+0.5*self.GeoTransform[1]+i*self.GeoTransform[1] for i in range(self.DataSet.RasterXSize)]),\
                                 index=np.array([self.GeoTransform[3]+0.5*self.GeoTransform[-1]+i*self.GeoTransform[-1] for i in range(self.DataSet.RasterYSize)]))
        self.DEM[self.DEM==-9999]= np.NaN
        self.DEM[self.DEM<0]= np.NaN
        if kwargs.get('ajustar',False): self.DEM = self.DEM.dropna(how='all').dropna(how='all',axis=1)

    def Setcolor(self,x, color):
        for m in x.keys():
            for t in x[m][1]:
                t.set_color(color)
    @property
    def name(self):
        '''Encuentra el offset de la estacion'''
        return self.mysql_query("select NombreEstacion from estaciones where codigo = %s"%self.codigo)[0][0]

    @property
    def longitud(self):
        '''Encuentra el la longitud de la estacion'''
        return float(self.mysql_query("select Longitude from estaciones where codigo = %s"%self.codigo)[0][0])

    @property
    def latitud(self):
        '''Encuentra la latitud'''
        return float(self.mysql_query("select Latitude from estaciones where codigo = %s"%self.codigo)[0][0])
    @property
    def municipio(self):
        '''Encuentra el municipio donde se encuentra ubicada la estacion'''
        return self.mysql_query("select Ciudad from estaciones where codigo = %s"%self.codigo)[0][0]

    @staticmethod
    def datetimeToString(DatetimeObject):
        return DatetimeObject.strftime('%Y-%m-%d %H:%M')

    def mysql_query(self,query,toPandas=True):
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
        self.mysql_settings()
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        df = pd.read_sql(sql,conn_db)
        conn_db.close()
        return df

class Nivel(SiataData):
    ruta = os.getcwd()
    def __init__(self,codigo=None,codigos=None):
        SiataData.__init__(self,codigo,codigos)
    @property
    def infost(self):
        estaciones =self.mysql_query("select * from estaciones where Red='nivel'")
        estaciones.columns = self.mysql_desc_table('estaciones')
        estaciones = estaciones.set_index('Codigo')
        estaciones.index = np.array(estaciones.index,int)
        estaciones['Latitude'] = np.array(estaciones['Latitude'],float)
        estaciones['Longitude'] = np.array(estaciones['Longitude'],float)
        estaciones = estaciones.append(self.infostMocoa())
        return estaciones
    @property
    def xSensor(self):
        return pd.Series.from_array(dict(zip([128,108,245,109,106,186,124,135,247,140,96,101,246,92,94,245,238,251,1014,1013,260,158,182,93,239,90,104,143,183,240,99,91,115,116,134,152,166,179,155,236,173,178,196,195,259,268,98,272,273],[8.0,4.23,11.66,3.5,17.0,5.75,12.6,3.08,3.0,24.0,4.2,0.8,4.2,1.3,12.11,11.66,12.54,4.1,6.88,8.74,21.0,3.87,2.45,31.17,6.4,11.6,2.55,4.66,2.8,1.5,21.0,18.95,1.55,8.21,2.5,3.0,2.0,8.0,5.5,14.0,2.0,1.5,29.8,3.92,2.42,4.4,5.0,5.74,3.18])
    )).loc[self.codigo]

    @property
    def offsetOld(self):
        '''Encuentra el offset de las estaciones de nivel'''
        return float(self.mysql_query("select offsetN from estaciones where codigo = %s"%self.codigo)[0][0])

    @property
    def offset(self):
        try:
            offset = self.offsetData.loc[self.codigo,'offset'].values[-1]
        except AttributeError:
            offset = self.offsetData.loc[self.codigo,'offset']
        return offset


    @property
    def sensor_type(self):
        '''Encuentra el offset de las estaciones de nivel'''
        return int(self.mysql_query("select N from estaciones where codigo = %s"%self.codigo)[0][0])

    def filter_level(self):
        positive,negative = self.filter_negative(self.data['rawLevel'])
        self.data['N'] = 0
        if negative.size<>0:
            self.data.loc[list(negative),'N'] = 1
        self.data['FP'] = 0
        self.data['FP'][self.data['rawLevel']>0.90*self.data['offset']] = 1
        self.data['F'] = 0
        self.data.loc[self.data[self.data['rawLevel'].isnull()].index,'F']=1
        self.data['Level'] = self.data[(self.data['N']==0)&(self.data['FP']==0)]['rawLevel'].reindex(self.data.index)
        if self.get_data_status == False:
            self.data['N'] = 0
            self.data['FP'] = 0
            self.data['F'] = 1

    def get_level(self,start,end):
        self.get_data(start,end)
        self.filter_level()
        return self.data['Level']

    def good_index(self):
        self.good_index = self.data[(self.data['goodDate']==1) & (self.data['goodDate']==1) & (self.data['positive']==1)]['rawLevel'].index
        return self.good_index

    def quality(self):
        self.positive = self.data[self.data['positive']==1].index.size/float(self.data.index.size)
        self.goodDate = self.data[self.data['goodDate']==1].index.size/float(self.data.index.size)
        self.gooQuality = self.data[self.data['calidad']==1].index.size/float(self.data.index.size)
        self.quality = self.good_index.size/float(self.data.index.size)
        return self.quality

    def filter_daily(self):
        fd =  self.data[['N','FP','F']].resample('D',how='sum').fillna(0)/14.40
        fd['codigo'] = self.codigo
        return fd.set_index('codigo',append=True).unstack().T.unstack(0)

    def filter_dailySt(self,codigos,start,end):
        for i,j in enumerate(codigos):
            level = Nivel(j)
            s = level.get_level(start,end)
            if i == 0:
                fd = level.filter_daily()
            else:
                fd = fd.append(level.filter_daily())
        fd.index = level.infost.loc[fd.index,'NombreEstacion'].values
        return fd.loc[fd.sum(axis=1).sort_values(ascending=False).index]

    def __repr__(self):
        '''string to recreate the object'''
        return "Nivel({})".format(self.codigo)

    def __str__(self):
        '''string to recreate the main information of the object'''
        return 'Nombre: {}\nRed: Nivel\nCodigo: {}\nLongitud:{}\nLatitud: {}\nMunicipio: {}'.format(self.name,self.codigo,self.longitud,self.latitud,self.municipio)

    def __add__(self,other):
        return [self.codigo] + [other.codigo]

    def __len__(self):
        pass
        #return len(self.fullname())


    def plot_plantilla(self,df,ruteSave):
        self.Plot_Mapa2(self,add_scatter=[df['Longitude'].values,df['Latitude'].values],\
                        Drainage='/media/nicolas/maso/Mario/shapes/nets/Puente_Gabino_1061/Puente_Gabino_1061',\
                        add_stations=map(lambda x:x.decode('utf-8'),df['NombreEstacion'].values),\
                        georef=[6.556,5.975,-75.725,-75.1255],clim=[1300,3400],fontsize=24,\
                        decimales=4,\
                        textcolor='w')

        self.m.readshapefile('/media/nicolas/maso/Mario/shapes/streams/169/169','drenaje',
                             color=self.colores_siata[-3],
                             linewidth=3.0,zorder=5)
        self.m.readshapefile('/media/nicolas/maso/Mario/shapes/AreaMetropolitana','area',
                             color=self.colores_siata[-1],
                             linewidth=3.0,zorder=5)
        if ruteSave is None:
            pass
        else:
            plt.savefig(ruteSave,format=ruteSave[-3:],bbox_inches='tight')

    def plot_riskMap(self,ruteMap,ruteRisk,ruteSave, width = 540,height = 816):
        locale.setlocale(locale.LC_TIME, ('es_co','utf-8'))
        reload (sys)
        sys.setdefaultencoding('utf8')
        imgTemp = StringIO()
        imgDoc  = canvas.Canvas(imgTemp)
        imagenPDF = PdfImageSIATA (ruteRisk , width = width, height = height)
        imagenPDF.drawOn (imgDoc, 840,0)
        imgDoc.save()
        page	= PdfFileReader(file(ruteMap,"rb")).getPage(0)
        overlay = PdfFileReader(StringIO(imgTemp.getvalue())).getPage(0)
        page.mergePage(overlay)
        output = PdfFileWriter()
        output.addPage(page)
        output.write(file(ruteSave ,"w"))

    def plot_Calidad(self,fd,ruteSave=None):
        df = fd.loc[fd.index[::-1]]
        index = []
        for i,j in zip(range(1,df.index.size+1)[::-1],df.index):
            index.append('%d - %s'%(i,j))
        df.index = index
        fig = plt.figure(figsize=(14,20))
        im = plt.imshow(df.values, interpolation='nearest', vmin=0, vmax=100, aspect='equal',cmap='Blues');
        ax = plt.gca();
        ax.set_yticks(np.arange(0, df.index.size, 1));
        ax.set_yticklabels(df.index,fontsize=14,color=[0.0078,0.227,0.26],ha = 'left');
        ax.set_xticks(np.arange(1, df.columns.size, 3), minor=False,);
        ax.set_xticks(np.arange(0,df.columns.size,1),minor=True)
        ax.set_yticks(np.arange(-.5, df.index.size, 1), minor=True);
        ax.set_xticklabels(df.columns.get_level_values(level=0).strftime('17/%m/%d')[::3],
                           fontsize=14,color=[0.0078,0.227,0.26],minor=False,position=(0,1.015));
        ax.set_xticklabels(df.columns.get_level_values(level=1),
                           fontsize=14,color=[0.0078,0.227,0.26],minor=True);
        plt.draw()
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width*1.05 for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        color = 'lightgrey'
        ax.text(-0.4,df.index.size+1.3,
                'REPORTE DE CALIDAD DE DATOS\n%s - %s'%(fd.columns.get_level_values(level=0)[0].strftime('%Y-%m-%d'),fd.columns.get_level_values(level=0)[-1].strftime('%Y-%m-%d')),
                fontsize=22)
        #colorbar
        cbaxes = fig.add_axes([-0.06, 0.92, 0.25, 0.015])
        cbaxes.tick_params(axis='x', colors=[0.0078,0.227,0.26])
        cbar = fig.colorbar(im,cax = cbaxes, orientation="horizontal")
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(u'% datos malos según criterio (N, FP, F)',fontsize=12)
        for i in np.arange(im.get_extent()[0],im.get_extent()[1]+1,3):
            ax.axvline(i, color=color, linestyle='-')
        if ruteSave is None:
            pass
        else:
            plt.savefig(ruteSave,format=ruteSave[-3:],bbox_inches='tight')

    def filter_alert(self,df):
        def locate(valor,codigo):
            dif = valor - np.array([0]+list(self.get_riskDf().loc[codigo].values))
            return np.argmin(dif[dif >= 0])

        df_max = df.groupby([lambda x:x.day]).max().T.dropna()
        #df_max = df_max.loc[self.df_info.loc[df_max.index].sort_values(by='Latitude').index]
        data = []
        for codigo in df_max.index:
            row = []
            for dia in df_max.columns:
                row.append(locate(df_max.loc[codigo,dia],codigo))
            data.append(row)
        df_index = pd.DataFrame(data,index=df_max.index,columns = df_max.columns)
        return df_index

    @staticmethod
    def plot_riskLevel(df,bbox_to_anchor = (-0.15, 1.09),ruteSave = None,legend=True):
        df = df.loc[df.index[::-1]]
        c = mcolors.ColorConverter().to_rgb
        cm = SiataData.make_colormap([c('#D9E5E8'),0.20,c('green'),0.4,c('orange'),0.60,c('red'),0.80,c('indigo')])
        fig = plt.figure(figsize=(6,14))
        im = plt.imshow(df.values, interpolation='nearest', vmin=0, vmax=4, aspect='equal',cmap=cm);
        #cbar = fig.colorbar(im)
        ax = plt.gca();
        ax.set_xticks(np.arange(0,df.columns.size, 1));
        ax.set_yticks(np.arange(0, df.index.size, 1));
        ax.set_xticklabels(df.columns,fontsize=14);
        ax.set_yticklabels(df.index,fontsize=14,ha = 'left');
        ax.set_xticks(np.arange(-.5, df.columns.size, 1), minor=True,);
        ax.set_yticks(np.arange(-.5, df.index.size, 1), minor=True);
        plt.draw()
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width*1.05 for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.text(-0.4,df.index.size+0.5,'NIVELES DE RIESGO\n %s - %s'%(start,pd.to_datetime(end).strftime('%Y-%m-%d')),fontsize=16)
        alpha=1
        height = 8
        if legend == True:
            p1 = Rectangle((0, 0), 1, height, fc="green",alpha=alpha)
            p2 = Rectangle((0, 0), 1, height, fc="orange",alpha=alpha)
            p3 = Rectangle((0, 0), 1, height, fc="red",alpha=alpha)
            p4 = Rectangle((0, 0), 1, height, fc="indigo",alpha=alpha)
            #p2 = Rectangle((0, 0), 1, 1, fc="red")
            leg = ax.legend([p1,p2,p3,p4], [u'N1',u'N2',u'N3',u'N4'],
                       ncol=3,bbox_to_anchor=bbox_to_anchor,fontsize=14)
        if ruteSave is None:
            pass
        else:
            plt.savefig(ruteSave,format='pdf',bbox_inches='tight')


    @staticmethod
    def get_level_stations(codigos,start,end):
        try:
            s = Nivel(codigos[0]).get_level(start,end)
            df = pd.DataFrame(s,columns=[codigos[0]])
        except:
            Index = pd.date_range(start,end,freq='min')
            df = pd.DataFrame([np.NaN]*Index.size,index = Index,columns=[codigos[0]])
            pass
        for i in codigos:
            level = Nivel(i)
            try:
                df[i] = level.get_level(start,end)
            except:
                df[i] = np.NaN
                pass
        return df
    def get_riskDf(self):
        '''obtiene los indices de las estaciones que tienen la informacion de riesgo actualizada'''
        dfr = self.infost[['minor_flooding','moderate_flooding','major_flooding','action_level']]
        dfr[dfr==-999]=np.NaN
        return dfr[['action_level','minor_flooding','moderate_flooding','major_flooding']]

    @staticmethod
    def indexToNames(codigos):
        return self.infost.loc[codigos,'NombreEstacion'].values

    @staticmethod
    def namesToIndex(names):
        return Nivel().infost.set_index('NombreEstacion',append=True).reset_index(0).loc[names]['level_0'].values

    def plot_CalidadDayly(self):
        end = pd.to_datetime((datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%Y-%m-%d 23:59'))
        start = end-datetime.timedelta(days=7)+datetime.timedelta(minutes=1)
        end,start = self.datetimeToString(end),start.strftime('%Y-%m-%d')
        fd = self.filter_dailySt(self.infost.index,start,end)
        nombre = 'reporteCalidad.pdf'
        self.plot_Calidad(fd,ruteSave=nombre)
        os.system('scp %s mcano@siata.gov.co:/var/www/mario/'%nombre)
        df = self.get_level_stations(self.namesToIndex(fd.index),start,end)
        df.columns = self.indexToNames(df.columns)
        df.plot(subplots=True,figsize=(14,90),grid=True)
        nombre = 'Niveles.pdf'
        plt.savefig(nombre,bbox_inches='tight')
        os.system('scp %s mcano@siata.gov.co:/var/www/mario/'%nombre)

    @property
    def firstDate(self):
        for i in self.mysql_query("select fecha from datos where cliente='%s' limit 2000"%self.codigo).fillna(np.NaN)[0].dropna().values:
            try:
                date = pd.to_datetime(i).strftime('%Y-%m-%m')
                break
            except:
                date = None
                pass
        return date

    def sort_byLatitude(self,codigos):
        return self.infost.loc[codigos].sort_values('Latitude').index

    def basin_set_DemDir(self,ruta_dem,ruta_dir,nodata=-9999.0,dxp=12.7):
        wmf.cu.nodata=nodata
        wmf.cu.dxp=dxp
        DEM = wmf.read_map_raster(ruta_dem,True)
        DIR = wmf.read_map_raster(ruta_dir,True)
        DIR[DIR<=0] = wmf.cu.nodata.astype(int)
        DIR = wmf.cu.dir_reclass_rwatershed(DIR,wmf.cu.ncols,wmf.cu.nrows)
        return DEM,DIR

    def basin_maker(self,lon,lat,dxp=12.7,add=0.001,add_out=0.07,dt =300,save=False,**kwargs):
            if dxp == 60:
                ruta_dem = '/media/nicolas/Home/nicolas/01_SIATA/raster/dem_amva60.tif'
                ruta_dir = '/media/nicolas/Home/nicolas/01_SIATA/raster/dir_amva60.tif'
                umbral = kwargs.get('umbral',1000)
                rute_st = '../information/stream60m.csv'
                rute_net = '/media/nicolas/maso/Mario/shapes/nets/Net60m/Net60m'
            else:
                ruta_dem = '/media/nicolas/Home/nicolas/01_SIATA/raster/dem_amva12.tif'
                ruta_dir = '/media/nicolas/Home/nicolas/01_SIATA/raster/dir12.tif'
                umbral = kwargs.get('umbral',500)
                rute_st = '../information/stream12m.csv'
                rute_net = '/media/nicolas/maso/Mario/shapes/nets/Net12m/Net12m'
            rutaShapes = '/media/nicolas/maso/Mario/shapes'
            wmf.cu.nodata=-9999.0
            wmf.cu.dxp=dxp
            llcrnrlat=lat-add;urcrnrlat=lat+add;llcrnrlon=lon-add;urcrnrlon=lon+add
            df = self.read_csv(rute_st)
            bt = df[((df.index.values>(llcrnrlon))&(df.index.values<(urcrnrlon)))&((df['Y']>llcrnrlat)&(df['Y']<urcrnrlat))]
            A = map(lambda x,y: [x,y],bt.index,bt.Y.values)
            lon_found,lat_found = tuple(A[spatial.KDTree(A).query((lon,lat))[1]])
            fig = plt.figure()
            axis =fig.add_subplot(111)
            m = Basemap(projection='merc',llcrnrlat=llcrnrlat-add_out,urcrnrlat=urcrnrlat+add_out,
                            llcrnrlon=llcrnrlon- add_out,urcrnrlon=urcrnrlon + add_out,resolution='c',ax=axis)
            m.readshapefile(rute_net,'net')
            x,y = m(lon,lat)
            x3,y3 = m(lon_found,lat_found)
            x2,y2 = m(bt.index.values,bt['Y'].values)
            m.scatter(x,y,s=100,zorder = 20)
            m.scatter(x2,y2,s=50,color='tan',alpha=0.4)
            m.scatter(x3,y3,s=50,color='r')
            DEM,DIR = self.basin_set_DemDir(ruta_dem,ruta_dir,-9999.0,dxp = dxp)
            st = wmf.Stream(lon_found,lat_found,DEM,DIR,name = 'Stream%s'%self.name)
            cu = wmf.SimuBasin(lon_found,lat_found, DEM, DIR,name='Basin%s'%self.codigo, dt = dt, umbral=umbral, stream=st)
            if save==True:
                print 'guardando cuenca'
                os.system('mkdir %s/streams/%s'%(rutaShapes,self.codigo))
                st.Save_Stream2Map('%s/streams/%s/%s.shp'%(rutaShapes,self.codigo,self.codigo))
                os.system('mkdir %s/basins/%s'%(rutaShapes,self.codigo))
                cu.Save_Basin2Map('%s/basins/%s/%s.shp'%(rutaShapes,self.codigo,self.codigo),dx=dxp)
                os.system('mkdir %s/nets/%s'%(rutaShapes,self.codigo))
                cu.Save_Net2Map('%s/nets/%s/%s.shp'%(rutaShapes,self.codigo,self.codigo))
                cu.set_Geomorphology()
                cu.GetGeo_Cell_Basics()
                cu.GetGeo_Parameters()
                cu.Save_SimuBasin('/media/nicolas/maso/Mario/basins/%s.nc'%self.codigo,ruta_dem = ruta_dem,ruta_dir = ruta_dir)
                df = self.read_csv('../information/basinMaker.csv')
                df.loc[self.codigo] = [lon_found,lat_found]
                df.to_csv('../information/basinMaker.csv')
                m.readshapefile('%s/basins/%s/%s'%(rutaShapes,self.codigo,self.codigo),'basin',zorder=30,color='w')
                patches = []
                for info, shape in zip(m.basin, m.basin):
                    patches.append( Polygon(np.array(shape), True))
                    axis.add_collection(PatchCollection(patches, facecolor= self.colores_siata[0], edgecolor=self.colores_siata[6], linewidths=0.5, zorder=30,alpha=0.6))

    def set_cu(self):
        self.cu = wmf.SimuBasin(rute='/media/nicolas/maso/Mario/basins/%s.nc'%self.codigo)

    @property
    def get_level_historicalData(self):
        print 'getting historical data\nWarning:It may take a couple of minutes'
        historicalLevel = self.get_level(self.firstDate,self.lastDate[0])
        historicalLevel.to_csv('/media/nicolas/maso/Mario/niveles_historicos/%s.csv'%self.codigo)
        return historicalLevel

    def read_level_historicalData(self):
        return pd.Series.from_csv('/media/nicolas/maso/Mario/niveles_historicos/%s.csv'%self.codigo)

    @property
    def offsetData(self):
        '''Tabla historico_bancallena_offset'''
        df = self.mysql_query('select * from historico_bancallena_offset;')
        df.columns = self.mysql_query('describe historico_bancallena_offset;')[0].values
        df[df==-9999]=np.NaN
        return df.set_index('codigo')

    def get_radarRain(self,codigo,start,end,**kwargs):
        '''Gets Radar Rain:
        kwargs are rute_nc, ruteRadar, ruteSave
        output:
        radar rain .bin .hdr'''
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        rute_nc = kwargs.get('rute_nc','/media/nicolas/maso/Mario/basins/%s.nc'%codigo)
        ruteRadar = kwargs.get('ruteRadar','/media/nicolas/Home/nicolas/101_RadarClass/')
        ruteSave = kwargs.get('ruteSave','/media/nicolas/maso/Mario/campos_lluvia/%s_%sto%s'%(codigo,start.strftime('%Y%m%d%H%M'),end.strftime('%Y%m%d%H%M')))
        dt = 300
        print '\nUsing basin from <<%s>>'%rute_nc
        print '\ndt = %s'%dt
        hora_inicial,hora_final = (start.strftime('%H:%M'),end.strftime('%H:%M'))
        query = '/home/nicolas/self_code/RadarConvStra2Basin2.py %s %s %s %s %s -t %s -v -s -1 %s -2 %s'%(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),rute_nc,ruteRadar,ruteSave,dt,hora_inicial,hora_final)
        print '\ngetting data from <<%s>>\n'%ruteRadar
        print 'Query:\n%s'%query
        output = os.system(query)
        if output == 0:
            print '\nRain saved in <<%s>>'%ruteSave
        else:
            print '\nWARNING: something went wrong'
        return query

    def read_rain(self,start=None,end=None,rute='Default'):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if rute=='Default':
            rute = '/media/nicolas/maso/Mario/campos_lluvia/%s_%sto%s'%(self.codigo,start.strftime('%Y%m%d%H%M'),end.strftime('%Y%m%d%H%M'))
        return wmf.read_mean_rain(rute+'.hdr')


    def rainPath(self,start,end):
        return '/media/nicolas/maso/Mario/campos_lluvia/%s_%sto%s'%(self.codigo,start.strftime('%Y%m%d%H%M'),end.strftime('%Y%m%d%H%M'))

    def rain_records(self,path,start,end):
        df = pd.read_csv('%s.hdr'%path,skiprows=range(5),index_col=3)
        df.index = df.index.to_datetime()
        return df.loc[start:end][df[' Record']<>1]

    def read_rainDate(self,date,path):
        df = pd.read_csv('%s.hdr'%path,skiprows=range(5),index_col=3)
        df.index = df.index.to_datetime()
        if df.loc[date,' Record'] == 1:
            return np.zeros(self.cu.ncells)
        else:
            return wmf.models.read_int_basin('%s.bin'%path,df.loc[date,' Record'],self.cu.ncells)[0]

    def read_rainVect(self,start,end):
        vector = []
        path = self.rainPath(pd.to_datetime(start),pd.to_datetime(end))
        index = self.rain_records(path,start,end).index
        for fecha in index:
                vector.append(self.read_rainDate(fecha,path))
        df = pd.DataFrame(vector,index=index)
        if df.index.size<3:
            df = pd.DataFrame(0,index = pd.date_range(start,end,freq='5min'),columns = range(self.cu.ncells))
        return df

    @property
    def cmap_rain(self):
        c = mcolors.ColorConverter().to_rgb
        ranges = np.array([  1,   5,  10,  20,  30,  40,  50,  65,  80, 100, 120])
        relations = ranges/float(max(ranges))
        colors = ['#00ffff','blue','lime','green','yellow','orange','darkorange','red','darkmagenta','violet']
        cdict = []
        for i,j in zip(colors,relations):
            cdict.append(j)
            cdict.append(c(i))
            cdict.append(c(i))
        cdict = cdict[1:-1]
        return self.make_colormap(cdict)

    def basin_mappable(self,extra_long=0,extra_lat=0):
        Mcols,Mrows=wmf.cu.basin_2map_find(self.cu.structure,self.cu.ncells)
        Map,mxll,myll=wmf.cu.basin_2map(self.cu.structure,self.cu.structure[0],Mcols,Mrows,self.cu.ncells)
        longs=np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(Mcols)])
        lats=np.array([myll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(Mrows)])
        X,Y=np.meshgrid(longs,lats)
        Y=Y[::-1]
        m = Basemap(projection='merc',llcrnrlat=lats.min()-extra_lat, urcrnrlat=lats.max()+extra_lat,
            llcrnrlon=longs.min()-extra_long, urcrnrlon=longs.max()+extra_long, resolution='c')
        return m

    def riskLegend(self,ax,alpha=0.5,**kwargs):
        levels = []
        for color in ['green','orange','red','indigo']:
            levels.append(Rectangle((0, 0), 1, 1, fc=color,alpha=alpha))
        return ax.legend(levels, [u'Nivel de alerta 1',u'Nivel de alerta 2',u'Nivel de alerta 3',u'Nivel de alerta 4'],**kwargs)

    def plot_level(self,level,rain,riesgos,fontsize=14,ncol=4,ax=None,bbox_to_anchor=(1.0,1.2),**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(13.,4))
            ax = fig.add_subplot(111)
        nivel = level.resample('H',how='mean')
        nivel.plot(ax=ax,label='',color='k')
        nivel.plot(alpha=0.3,label='',color='r',fontsize=fontsize,**kwargs)
        axu= ax.twinx()
        axu.set_ylabel('Lluvia promedio [mm/h]',fontsize=fontsize)
        mean_rain = rain.resample('H',how='sum')
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
        self.riskLegend(ax,ncol=4,bbox_to_anchor=(1.0,1.2))
        return (ax,axu)

    def plot_curvasDuracion(self,date,ax=None,lw=2.0,fontsize=14):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        wls = self.get_riskDf().loc[self.codigo].values
        df = self.read_level_historicalData().resample('H',how='mean')
        x,y = self.curva_duracion(df.dropna())
        ax.plot(x,y,color=self.colores_siata[-1],label=u'Hist.',linewidth=lw)
        ax.set_xlabel('Probabilidad de ser excedido [%]',fontsize=fontsize)
        level30 = self.get_level(pd.to_datetime(date)-datetime.timedelta(days=30),date).resample('H',how='mean')
        x,y = self.curva_duracion(level30.dropna())
        ax.plot(x,y,label=u'Últ.30 días',linestyle='-.',linewidth=lw,color='b')
        df = level30.loc[pd.to_datetime(date)-datetime.timedelta(days=7):date].resample('H',how='mean')
        x,y = self.curva_duracion(df.dropna())
        ax.plot(x,y,color='r',label=u'Últ.7 días',linestyle='--',linewidth=lw)
        #ax3.set_xlabel('Probabilidad de ser excedido [%]',fontsize=14)
        ax.legend(ncol=3,bbox_to_anchor=(1,1.15))
        ax.set_ylabel('Nivel (cm)',fontsize=fontsize)
        x = np.arange(0,100,0.1)
        ax.fill_between(x,ax.get_ylim()[0],wls[0],alpha=0.1,color=self.colores_siata[0])
        alpha=0.2
        ax.fill_between(x,wls[0],wls[1],alpha=alpha,color='green')
        ax.fill_between(x,wls[1],wls[2],alpha=alpha,color='orange')
        ax.fill_between(x,wls[2],wls[3],alpha=alpha,color='red')
        ax.fill_between(x,wls[3],ax.get_ylim()[1],alpha=alpha,color='indigo')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.grid()

    def update_estaciones(self,field,value):
        string = "%s = '%s'"%(field,value)
        return self.execute_mysql("UPDATE estaciones SET %s where codigo = '%s';"%(string,self.codigo))

    def execute_mysql(self,execution):
        print 'EXECUTION:\n %s'%execution
        self.mysql_settings()
        conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
        db_cursor = conn_db.cursor ()
        try:
            db_cursor.execute(execution)
            conn_db.commit()
            conn_db.close ()
            status = 'worked'
        except:
            conn_db.close ()
            status = 'failed'
            pass
        print 'EXECUTION STATUS: %s \n'%status

    def basinMappable(self,vec=None,Min=None,Max=None,ruta=None,axis = None,extra_long=0,extra_lat=0,*args,**kwargs):
        Mcols,Mrows=wmf.cu.basin_2map_find(self.cu.structure,self.cu.ncells)
        Map,mxll,myll=wmf.cu.basin_2map(self.cu.structure,self.cu.structure[0],Mcols,Mrows,self.cu.ncells)
        longs=np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(Mcols)])
        lats=np.array([myll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(Mrows)])
        X,Y=np.meshgrid(longs,lats)
        Y=Y[::-1]
        m = Basemap(projection='merc',
            llcrnrlat=lats.min()-extra_lat,
            urcrnrlat=lats.max()+extra_lat,
            llcrnrlon=longs.min()-extra_long,
            urcrnrlon=longs.max()+extra_long,
            resolution='c',ax=axis)
        Xm,Ym=m(X,Y)
        nperim = wmf.cu.basin_perim_find(self.cu.structure,self.cu.ncells)
        perim = wmf.cu.basin_perim_cut(nperim)
        xp,yp=m(perim[0],perim[1])
        m.plot(xp, yp,color=tuple(self.colores_siata[-1]))
        return m

    def Plot_basin(self,vec=None,Min=None,
            Max=None,ruta=None,figsize=(10,7),
            ZeroAsNaN = 'no',extra_lat=0.0,extra_long=0.0,lines_spaces=0.02,
            xy=None,xycolor='b',colorTable=None,alpha=1.0,vmin=None,vmax=None,
            colorbar=True, colorbarLabel = None,axis=None,rutaShp=None,shpWidth = 0.7,
            shpColor = 'r', EPSG = 4326,backMap = False,
            **kwargs):
            #Plotea en la terminal como mapa un vector de la cuenca
            #Prop de la barra de colores
            cbar_ticklabels = kwargs.get('cbar_ticklabels', None)
            cbar_ticks = kwargs.get('cbar_ticks', None)
            cbar_ticksize = kwargs.get('cbar_ticksize', 14)
            show = kwargs.get('show', True)
            ShpIsPolygon = kwargs.get('ShpIsPolygon',None)
            shpAlpha = kwargs.get('shpAlpha',0.5)
            xy_colorbar = kwargs.get('xy_colorbar', False)
            #El mapa
            Mcols,Mrows=wmf.cu.basin_2map_find(self.cu.structure,self.cu.ncells)
            Map,mxll,myll=wmf.cu.basin_2map(self.cu.structure,self.cu.structure[0]
                ,Mcols,Mrows,self.cu.ncells)
            longs=np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(Mcols)])
            lats=np.array([myll+0.5*wmf.cu.dy+i*wmf.cu.dy for i in range(Mrows)])
            X,Y=np.meshgrid(longs,lats)
            Y=Y[::-1]
            show = kwargs.get('show',True)
            if axis is None:
                fig = pl.figure(figsize = figsize)
                ax = fig.add_subplot(111)
            else:
                show = False
            m = Basemap(projection='merc',
                llcrnrlat=lats.min()-extra_lat,
                urcrnrlat=lats.max()+extra_lat,
                llcrnrlon=longs.min()-extra_long,
                urcrnrlon=longs.max()+extra_long,
                resolution='c',
                epsg = EPSG,ax=axis)
            Xm,Ym=m(X,Y)
            #Plotea el contorno de la cuenca y la red
            nperim = wmf.cu.basin_perim_find(self.cu.structure,self.cu.ncells)
            perim = wmf.cu.basin_perim_cut(nperim)
            xp,yp=m(perim[0],perim[1])
            per_color = kwargs.get('per_color',typColor)
            per_lw = kwargs.get('per_lw',2)
            m.plot(xp, yp, color=per_color,lw=per_lw)
            #hay una variable la monta
            if vec is not None:
                if vmin is None:
                    vmin = vec.min()
                if vmax is None:
                    vmax = vec.max()
                MapVec,mxll,myll=wmf.cu.basin_2map(self.cu.structure,vec,Mcols,Mrows,
                    self.cu.ncells)
                MapVec[MapVec==wmf.cu.nodata]=np.nan
                if ZeroAsNaN is 'si':
                    MapVec[MapVec == 0] = np.nan
                if colorTable is not None:
                    cs = m.contourf(Xm, Ym, MapVec.T, 25, alpha=alpha,cmap=colorTable,
                        vmin=vmin,vmax=vmax)
                else:
                    cs = m.contourf(Xm, Ym, MapVec.T, 25, alpha=alpha,
                        vmin=vmin,vmax=vmax)
                cbar_label_size = kwargs.get('cbar_label_size',16)
            else:
                cs = None
            #Si hay coordenadas de algo las plotea
            #Guarda
            return m,cs

    def plot_rain(self,vec,extra_long=0.0,extra_lat = 0.0,ax = None,net=True,labels = [True,False,False,True],coordinates=True,cbarTitle=False,**kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        vec[0],vec[1],vec[2] = (120.,100.,80.)
        vec[vec>120] = 120
        m,cs = self.Plot_basin(vec=vec,colorTable=self.cmap_rain,axis=ax,vmin=0,vmax=120,extra_long=extra_long,extra_lat=extra_lat)
        x,y = m(self.longitud,self.latitud)
        m.scatter(x,y,color='grey',s=80)
        cbar_ticks =[0, 1, 5, 10, 20, 30, 40, 50, 65, 80, 100, 120]
        cbar_xtickslabels = ['<5']+map(lambda x:str(x),cbar_ticks[1:-1])+['>100']
        cbar = m.colorbar(cs,ticks = cbar_ticks,location='bottom')
        cbar.set_ticklabels(cbar_xtickslabels)
        if net:
            m.readshapefile('/media/nicolas/maso/Mario/shapes/nets/%s/%s'%(self.codigo,self.codigo),'net',color='grey')
            m.readshapefile('/media/nicolas/maso/Mario/shapes/streams/%s/%s'%(self.codigo,self.codigo),'streams',color='grey',linewidth=2)
        if coordinates:
            self.map_coordinates(m)
            # labels = [left,right,top,bottom]
            meridians = m.drawmeridians(self.meridians,labels = labels,fmt='%.2f',color='w')
            parallels = m.drawparallels(self.parallels,labels = labels,fmt='%.2f',color='w')
            for i in parallels:
                try:
                    parallels[i][1][0].set_rotation(90)
                except:
                    pass
        if cbarTitle <> False:
            cbar.ax.set_xlabel(cbarTitle,color=self.colores_siata[-1])
        # 666 - CBAR REMOVED
        cbar.remove()
        return m


    def get_radarRainV2(self,codigo,start,end,**kwargs):
        '''Gets Radar Rain:
        kwargs are rute_nc, ruteRadar, ruteSave
        output:
        radar rain .bin .hdr'''
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)+datetime.timedelta(hours=5) #cambio utc
        rute_nc = kwargs.get('rute_nc','/media/nicolas/maso/Mario/basins/%s.nc'%codigo)
        ruteRadar = kwargs.get('ruteRadar','/media/nicolas/Home/nicolas/101_RadarClass/')
        ruteSave = kwargs.get('ruteSave','/media/nicolas/maso/Mario/campos_lluvia/%s_%sto%s'%(codigo,start.strftime('%Y%m%d%H%M'),(end-datetime.timedelta(hours=5)).strftime('%Y%m%d%H%M')))
        dt = 300
        print '\nUsing basin from <<%s>>'%rute_nc
        print '\ndt = %s'%dt
        hora_inicial,hora_final = (start.strftime('%H:%M'),end.strftime('%H:%M'))
        query = '/home/nicolas/self_code/RadarConvStra2Basin3.py %s %s %s %s %s -t %s -v -s -1 %s -2 %s'%(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),rute_nc,ruteRadar,ruteSave,dt,hora_inicial,hora_final)
        print '\ngetting data from <<%s>>\n'%ruteRadar
        print 'Query:\n%s'%query
        r = os.system(query)
        if r <> 0:
            print 'Warning: Something went wrong'

    @property
    def rainDates(self):
        return sorted(map(lambda x:'%s-%s-%s %s:%s'%(x[43:47],x[47:49],x[49:51],x[51:53],x[53:55]), glob.glob('/media/nicolas/Home/nicolas/101_RadarClass/*.nc')))


    def current_level(self,minutes=5):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(minutes=minutes)
        query = "select DATE_FORMAT(fecha,%s),DATE_FORMAT(hora, %s),pr,NI,calidad from datos \
                where cliente = '%s' and fecha >= '%s' and hora >='%s';"%("'%Y-%m-%d'","'%H:%i:%s'",self.codigo,start.strftime('%Y-%m-%d'),start.strftime('%H:%M'))
        return self.get_level(self.datetimeToString(start),self.datetimeToString(end),query=query)

    def update_offsetData(self,field,value,fecha_hora):
        string = "%s = '%s'"%(field,value)
        return self.execute_mysql("UPDATE historico_bancallena_offset SET %s where codigo = '%s' and fecha_hora = '%s';"%(string,self.codigo,fecha_hora))

    def insert_offsetData(self,values):
        self.execute_mysql('INSERT INTO historico_bancallena_offset (codigo,fecha_hora,fecha,hora,descripcion,bancallena,offset,flag_ni,flag_pr,calidad,estado,fechaUltimaModificacion,bancamasllena) VALUES (%s)'%str(values).strip('[]'))

    @property
    def informacion(self):
        return self.infost.loc[187]

    def insert_offsetDataEjemplo(self):
        return [195,'2017-05-05 00:00:00','2017-05-05','00:00:00','actualizacion de estaciones faltantes',200,286,0,1,1,'A','2017-05-05 00:00:00',340]

    def offsetList(self,series):
        for i in self.mysql_query("select fecha_hora,offset from historico_bancallena_offset where codigo = '%s';"%self.codigo,toPandas=False):
            series.loc[i[0]:] = i[1]
        return series

    def lastDates(self,limit = 15):
        formats = ("'%Y-%m-%d'","'%H:%i:%s'",['pr','NI'][self.sensor_type],self.codigo,limit)
        query = "SELECT DATE_FORMAT(fecha,%s),DATE_FORMAT(hora, %s),%s FROM datos \
                     WHERE cliente = '%s' and fecha < '2020' ORDER BY fecha DESC LIMIT %s;"%formats
        return query

    @property
    def lastDate(self):
        return self.mysql_query(self.lastDates(limit=1),toPandas=False)[0][:2]

    def level_query(self,start,end):
        inicia,finaliza = pd.to_datetime(start).strftime('%Y-%m-%d'),pd.to_datetime(end).strftime('%Y-%m-%d')
        formats = ("'%Y-%m-%d'","'%H:%i:%s'",['pr','NI'][self.sensor_type],self.codigo,inicia,finaliza)
        if inicia == finaliza:
            query = "SELECT DATE_FORMAT(fecha,%s),DATE_FORMAT(hora, %s),%s FROM datos \
                     WHERE cliente = '%s' and fecha = '%s';"%formats[:-1]
        else:
            query = "SELECT DATE_FORMAT(fecha,%s),DATE_FORMAT(hora, %s),%s FROM datos \
                     WHERE cliente = '%s' and fecha BETWEEN '%s' and '%s';"%formats
        return query

    def get_data(self,start,end,**kwargs):
        query = kwargs.get('query',self.level_query(start,end))
        nivel = self.mysql_query(query)
        if nivel.values.size == 0:
            self.get_data_status = False
            print 'Warning: No data found for station << %s - %d >>, output will be a DataFrame filled with NaN'%(self.name,self.codigo)
            df = pd.DataFrame(index=pd.to_datetime(self.datetimeToString(pd.date_range(start,end,freq='min'))),columns=[u'sensor', u'offset', u'rawLevel'])
            self.data = df
        else:
            self.get_data_status=True
            lista = []
            flag=0
            for i,j in zip(nivel.apply(lambda x:'%s %s'%(x[0],x[1][:-3]),axis=1).values,nivel[2].values):
                if i == flag:
                    pass
                else:
                    try:
                        lista.append([pd.to_datetime(i),float(j)])
                        flag=i
                    except:
                        pass
            df = pd.DataFrame(lista).set_index(0)
            df = df.reindex(pd.to_datetime(self.datetimeToString(pd.date_range(start,end,freq='min'))))
            df.columns = ['sensor']
            df['offset']= np.NaN
            self.offsetList(df['offset'])
            df['rawLevel'] = df['offset']-df['sensor']
            self.data = df.loc[start:end]
        return df

    def get_30daysLevel(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=30)
        for i in self.infost.index:
            nivel = Nivel(i)
            nivel.get_data(start,end)
            nivel.filter_level()
            nivel.data.to_csv('/media/nicolas/maso/Mario/30daysLevel/%s_30days.csv'%i)
        print 'last month data stored'

    def update_30daysLevel(self):
        ruta = '/media/nicolas/maso/Mario/30daysLevel/%s_30days.csv'%self.codigo
        df = self.read_csv(ruta)
        end = datetime.datetime.now()
        start = end-datetime.timedelta(minutes = 5)
        datef1 = "DATE_FORMAT(fecha,'%Y-%m-%d')"
        datef2 = "DATE_FORMAT(hora,'%H:%i:%s')"
        query = "SELECT %s,%s,%s FROM datos WHERE cliente = '%s' and (((fecha>'%s') or (fecha = '%s' and hora>='%s')) and ((fecha<'%s') or (fecha = '%s' and hora<= '%s')))"%(datef1,datef2,['pr','NI'][self.sensor_type],self.codigo,start.strftime('%Y-%m-%d'),start.strftime('%Y-%m-%d'),start.strftime('%H:%M:00'),end.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),end.strftime('%H:%M:00'))
        self.get_data(start,end,query=query)
        self.filter_level()
        df = df.append(self.data)
        df.drop(df.index[:self.data.index.size])
        df.to_csv(ruta)


    def update_30daysAll(self):
        print 'updating'
        codigos = self.infost[self.infost['estado']=='A'].index
        for i in codigos:
            try:
                nivel = Nivel(i)
                nivel.update_30daysLevel()
            except:
                'something wrong with %s'%i

    @property
    def level30Days(self):
        ruta = '/media/nicolas/maso/Mario/30daysLevel/%s_30days.csv'%self.codigo
        return self.read_csv(ruta)
    @property
    def riskLevels(self):
        return self.mysql_query("SELECT action_level, minor_flooding, moderate_flooding, major_flooding FROM estaciones WHERE codigo = '%s'"%self.codigo,toPandas=False)[0]

    def get_level_6months(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(months=6)
        for i in self.infost.index:
            nivel = sd.Nivel(i)
            nivel.get_data(start,end)
            nivel.filter_level()
            nivel.data.to_csv('/media/nicolas/maso/Mario/120daysLevel/%s_120days.csv'%i)
            print nivel.name
    @property
    def level120Days(self):
        ruta = '/media/nicolas/maso/Mario/120daysLevel/%s_120days.csv'%self.codigo
        return self.read_csv(ruta)

    def plot_levelRisk(self,nivel,riesgos,fontsize=16,ncol=4,ax=None,bbox_to_anchor=(1.0,1.2),ruteSave=False,legend=True,**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(15.,4))
            ax = fig.add_subplot(111)
        nivel.index = pd.to_datetime(nivel.index)
        nivel.plot(alpha=0.2,label='',color='r',ax = ax,fontsize=fontsize,**kwargs)
        nivel = nivel.resample('5min',how='mean')
        nivel.plot(ax=ax,label='',color='k',alpha=1.0,lw=2,fontsize=fontsize)
        ax.set_ylabel('Nivel (cm)',fontsize=fontsize)
        alpha=0.2
        ax.fill_between(nivel.index,ax.get_ylim()[0],riesgos[0],alpha=0.1,color=self.colores_siata[0])
        ax.fill_between(nivel.index,riesgos[0],riesgos[1],alpha=alpha,color='green')
        ax.fill_between(nivel.index,riesgos[1],riesgos[2],alpha=alpha,color='orange')
        ax.fill_between(nivel.index,riesgos[2],riesgos[3],alpha=alpha,color='red')
        ax.fill_between(nivel.index,riesgos[3],max(riesgos)*1.05,alpha=alpha,color='indigo')
        ax.set_ylim(nivel.min()*0.7,max(riesgos)*1.05)
        #ax.set_title(u'Código: %s - Nombre: %s'%(self.codigo,self.infost.loc[self.codigo,'NombreEstacion']),fontsize=18,color = self.colores_siata[-1])
        if legend == True:
            leg = self.riskLegend(ax,ncol=4,bbox_to_anchor=bbox_to_anchor,fontsize=fontsize)
            for text in leg.get_texts():
                plt.setp(text, color = self.colores_siata[-2])
        if ruteSave <> False:
            plt.savefig(ruteSave,bbox_inches='tight')
    def plot_level(self,level,rain,riesgos,fontsize=14,ncol=4,ax=None,bbox_to_anchor=(1.0,1.2),**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(13.,4))
            ax = fig.add_subplot(111)
        nivel = level.resample('H',how='mean')
        nivel.plot(ax=ax,label='',color='k')
        nivel.plot(alpha=0.3,label='',color='r',fontsize=fontsize,**kwargs)
        axu= ax.twinx()
        axu.set_ylabel('Lluvia promedio [mm/h]',fontsize=fontsize)
        mean_rain = rain.resample('H',how='sum')
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
        self.riskLegend(ax,ncol=4,bbox_to_anchor=(1.0,1.2))
        return (ax,axu)

    @property
    def riskNames(self):
        return ['action_level','minor_flooding','moderate_flooding','major_flooding']

    def update_riskData(self,riesgos):
        for tableName,riskValue in zip(self.riskNames,riesgos):
            self.update_estaciones(tableName,riskValue)

    def locateRisk(self,value):
        dif = value - np.array([0]+list(self.riskLevels))
        return np.argmin(dif[dif >= 0])

    def legendText(self,level):
        maxRisk = ['Nivel seguro','Nivel de alerta 1','Nivel de alerta 2','Nivel de alerta 3','Nivel de alerta 4'][self.locateRisk(level.max())]
        goodData = '%.1f'%(level.dropna().index.size*100.0/level.index.size)
        text = u"Estación de Nivel tipo {}\nResolución temporal: 1 minutos\n% de datos transmitidos: {}\nProfundidad máxima: {} [cm]\nNivel de riesgo máximo: {}\nProfundidad promedio: {} [cm]\n*Calidad de datos aún\n sin verificar exhaustivamente".format(['Ultrasonido','Radar'][self.sensor_type], goodData, '%.1f'%level.max(), maxRisk, '%.1f'%level.mean())
        return text

    def plot_levelInfo(self,nivel,riesgos,fontsize=16,ncol=4,ruteSave=False,bbox_to_anchor=(1.0,-0.1),**kwargs):
        riesgos = self.riskLevels
        fontsize=16
        ncol=4
        kwargs = {}
        ruteSave = False
        fig = plt.figure(figsize=(15,12))
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
                              wspace=None, hspace=10)
        gs = gridspec.GridSpec(10, 20)
        factor = 6
        ax= plt.subplot(gs[:factor, :])
        ax2 = plt.subplot(gs[factor:, :])
        nivel.index = pd.to_datetime(nivel.index)
        nivel.plot(alpha=0.2,label='',color='r',ax = ax,fontsize=fontsize,**kwargs)
        nivel = nivel.resample('5min',how='mean')
        nivel.plot(ax=ax,label='',color='k',alpha=1.0,lw=2,fontsize=fontsize)
        ax.set_ylabel('Nivel (cm)',fontsize=fontsize)
        alpha=0.2

        ax.fill_between(nivel.index,ax.get_ylim()[0],riesgos[0],alpha=0.1,color=self.colores_siata[0])
        ax.fill_between(nivel.index,riesgos[0],riesgos[1],alpha=alpha,color='green')
        ax.fill_between(nivel.index,riesgos[1],riesgos[2],alpha=alpha,color='orange')
        ax.fill_between(nivel.index,riesgos[2],riesgos[3],alpha=alpha,color='red')
        ax.fill_between(nivel.index,riesgos[3],max(riesgos)*1.05,alpha=alpha,color='indigo')
        ax.set_ylim(nivel.min()*0.7,max(riesgos)*1.05)
        ax.set_title(u'Código: %s - Nombre: %s'%(self.codigo,self.infost.loc[self.codigo,'NombreEstacion'].decode('utf-8',errors='replace')),fontsize=18,color = self.colores_siata[-1])
        leg = self.riskLegend(ax,ncol=4,bbox_to_anchor=bbox_to_anchor,fontsize=fontsize)
        for text in leg.get_texts():
            plt.setp(text, color = self.colores_siata[-2])

        if ruteSave <> False:
            plt.savefig(ruteSave,bbox_inches='tight')

        ax2.text(0.0, 0.0, self.legendText(nivel), color=self.colores_siata[-2],fontsize=18,linespacing=1.4)
        ax2.text(0.0,0.8,'RESUMEN',color = self.colores_siata[-1],fontsize=18,)
        plt.axis('off')

    def map_coordinates(self,basemap,factor = 0.2,**kwargs):
        self.lonmax = basemap.boundarylonmax
        self.lonmin = basemap.boundarylonmin
        self.latmin = basemap.boundarylats[0]
        self.latmax = basemap.boundarylats[1]
        self.meridians = [self.lonmin+factor*(self.lonmax-self.lonmin),self.lonmin+(1-factor)*(self.lonmax-self.lonmin)]
        self.parallels = [self.latmin+factor*(self.latmax-self.latmin),self.latmin+(1-factor)*(self.latmax-self.latmin)]


    def adjust_basin(self,rel=0.766,fac=0.0):
        self.map_coordinates(self.Plot_basin(extra_long=fac,extra_lat=fac)[0])
        y = self.latmax-self.latmin
        x = self.lonmax-self.lonmin
        if x>y:
            extra_long = 0
            extra_lat = (rel*x-y)/2.0
        else:
            extra_lat=0
            extra_long = (y/(2.0*rel))-(x/2.0)
        self.map_coordinates(self.Plot_basin(extra_long=extra_long+fac,extra_lat=extra_lat+fac)[0])
        return extra_lat+fac,extra_long+fac

    def plantilla(self,fontsize=14,figsize=(16,11.8),*adjustKeys):
        plt.rc('font', **{'size'   : 14})
        fig = plt.figure(figsize=(16,11.8))
        #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9,wspace=0.9,hspace=0.4,**adjustKeys)
        gs = gridspec.GridSpec(2, 20)
        ax1 = plt.subplot(gs[0, :10])
        ax2 = plt.subplot(gs[0, 10:])
        ax3 = plt.subplot(gs[1,:])
        final = time.time()
        return (ax1,ax2,ax3)

    def vectRaincmap(self,vec):
        for pos,i in enumerate([1]+np.arange(0,130.,4)):
            vec[pos] = i


    def cu_extraCoord(self,rel=0.766):
        self.map_coordinates(self.Plot_basin()[0])
        fac = 0.1*(self.lonmax-self.lonmin)
        extra_lat,extra_long = self.adjust_basin(rel,fac)
        return extra_lat,extra_long

    def infostMocoa(self):
        estaciones =self.mysql_query("select * from estaciones where Red='mocoa-nivel' and estado='A'")
        estaciones.columns = self.mysql_desc_table('estaciones')
        estaciones = estaciones.set_index('Codigo')
        estaciones.index = np.array(estaciones.index,int)
        estaciones['Latitude'] = np.array(estaciones['Latitude'],float)
        estaciones['Longitude'] = np.array(estaciones['Longitude'],float)
        return estaciones


    def rainReport(self):
        # DATA
        now = datetime.datetime.now()
        start = self.datetimeToString(now-datetime.timedelta(hours=3)) # 3 horas atras
        end = self.datetimeToString(now+datetime.timedelta(hours=1)) # 1 hora en el futuro
        self.update_30daysLevel()
        self.set_cu()
        minutes = 180
        level = self.level30Days.iloc[-minutes:]['Level']
        level.index = pd.to_datetime(level.index)# now - 10 minutos
        level = level.reindex(pd.date_range(level.index[0],end,freq='min'))
        self.get_radarRainV2(self.codigo,start,end)
        self.set_cu()
        # REPORLAB
        nombre_archivo = ruteSave[:-3]+'pdf'
        pdf = canvas.Canvas(nombre_archivo,pagesize=(900,1100))
        cx = 0
        cy = 900
        pdf.drawImage(ruteSave,20,250,width=860,height=650)
        pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/pie.png',0,0,width=905,height=145.451)
        pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/cabeza.png',0,920,width=905,height=180)
        pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
        pdf.setFont("AvenirBook", 20)
        pdf.drawString(240,945,u'Estación %s - %s'%(self.name,self.datetimeToString(now)))
        pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/nivelActual.png',623,550,width=120,height=25)
        pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/legendRisk.png',67,170,width=770,height=70)
        pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/acumuladoLegend.jpg',807,600,width=60,height=275)

        pdf.setFont("AvenirBook", 15)
        pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
        pdf.drawString(100,180,u'Nivel sin riesgo')
        for i,j in zip(range(1,5),np.linspace(250,700,4)):
            pdf.drawString(j,180,u'Nivel de riesgo %s'%i)
        pdf.showPage()
        pdf.save()
        os.system('scp %s mcano@siata.gov.co:/var/www/mario/rainReport/%d'%(ruteSave[:-3]+'pdf',self.codigo))
        # LOG
        df = pd.DataFrame(columns = ['now','start','end'])
        df.loc[self.codigo] = [now,start,end]
        ruteLog = '/media/nicolas/Home/Jupyter/MarioLoco/rainReport/log.csv'
        self.read_csv(ruteLog).append(df).to_csv(ruteLog)

    def reportLog(self,rute='../rainReport/log.csv'):
        return self.read_csv(rute)

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

    def plot_Section(self,levantamiento,level):
        fig = plt.figure(figsize=(14,6))
        ax = fig.add_subplot(111)
        ax.plot(levantamiento['x'],levantamiento['y'])
        hline = ((levantamiento['x'].min()*1.1,level),(levantamiento['x'].max()*1.1,level))
        lev = pd.DataFrame.copy(levantamiento)
        if lev.iloc[0]['y']<level:
            lev = pd.DataFrame(np.matrix([lev.iloc[0]['x'],level]),columns=['x','y']).append(lev)
        if lev.iloc[-1]['y']<level:
            lev = lev.append(pd.DataFrame(np.matrix([lev.iloc[-1]['x'],level]),columns=['x','y']))
        condition = (lev['y']>=level).values
        flag = condition[0]
        nlev = []
        count = 0
        points = []
        for i,j in enumerate(condition):
            if j==flag:
                nlev.append(list(lev.iloc[i].values))
            else:
                count += 1
                line = (list(lev.iloc[i-1].values),list(lev.iloc[i].values)) #  #puntoA
                inter = self.line_intersection(line,hline)
                print 'intersection: (%s,%s)\n'%inter
                nlev.append(inter)
                nlev.append(list(lev.iloc[i].values))
                points.append(i)
            flag = j
            if count == 2:
                df = pd.DataFrame(np.matrix(nlev),columns = ['x','y'])
                df = df.iloc[points[0]:points[1]+2]


                ax.plot(levantamiento['x'].values,levantamiento['y'].values,color='k')
                df.plot(x='x',y='y',kind='scatter',ax=ax)
                plt.fill_between(df['x'].values,level,df['y'].values,alpha=0.2)
                count =0
                break
        return ax

    def lastBat(self,xSensor):
        dfl = self.mysql_query('select * from levantamiento_aforo_nueva')
        dfl.columns = self.mysql_query('describe levantamiento_aforo_nueva')[0].values
        dfl = dfl.set_index('id_aforo')
        for id_aforo in list(set(dfl.index)):
            id_estacion_asociada,fecha = self.mysql_query("SELECT id_estacion_asociada,fecha from aforo_nueva where id_aforo = %s"%id_aforo,toPandas=False)[0]
            dfl.loc[id_aforo,'id_estacion_asociada'] = int(id_estacion_asociada)
            dfl.loc[id_aforo,'fecha'] = fecha
        dfl = dfl.reset_index().set_index('id_estacion_asociada')
        lev = dfl[dfl['fecha']==max(list(set(pd.to_datetime(dfl.loc[self.codigo,'fecha'].values))))][['x','y']]
        level = self.level30Days.dropna()['Level'].max()
        cond = (lev['x']<xSensor).values
        flag = cond[0]
        for i,j in enumerate(cond):
            if j==flag:
                pass
            else:
                point = (tuple(lev.iloc[i-1].values),tuple(lev.iloc[i].values))
            flag = j

        intersection = self.line_intersection(point,((xSensor,0.1*lev['y'].min()),(xSensor,1.1*lev['y'].max(),(xSensor,))))
        lev = lev.append(pd.DataFrame(np.matrix(intersection),index=['xSensor'],columns=['x','y'])).sort_values('x')
        lev['y'] = lev['y']-intersection[1]
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

    def plot_section(self,df,*args,**kwargs):
        '''Grafica de la seccion transversal de estaciones de nivel
        |  ----------Parametros
        |  df : dataFrame con el levantamiento topo-batimetrico, columns=['x','y']
        |  level : Nivel del agua
        |  riskLevels : Niveles de alerta
        |  *args : argumentos plt.plot()
        |  **kwargs : xSensor,offset,riskLevels,xLabel,yLabel,ax,groundColor,fontsize,figsize,
        |  Nota: todas las unidades en metros'''
        # Kwargs
        level = kwargs.get('level',None)
        xLabel = kwargs.get('xLabel','Distancia desde la margen izquierda [m]')
        yLabel = kwargs.get('yLabel','Profundidad [m]')
        waterColor = kwargs.get('waterColor',self.colores_siata[0])
        groundColor = kwargs.get('groundColor',self.colores_siata[-2])
        fontsize= kwargs.get('fontsize',14)
        figsize = kwargs.get('figsize',(10,4))
        riskLevels = kwargs.get('riskLevels',None)
        xSensor = kwargs.get('xSensor',None)
        offset = kwargs.get('offset',self.offset)
        scatterSize = kwargs.get('scatterSize',0.0)
        ax = kwargs.get('ax',None)
        # main plot
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        ax.plot(df['x'].values,df['y'].values,color='k',lw=0.5)
        ax.fill_between(np.array(df['x'].values,float),np.array(df['y'].values,float),float(df['y'].min()),color=groundColor,alpha=0.2)
        # waterLevel
        if level is not None:
            for data in self.get_sections(df,level):
                ax.fill_between(data['x'],level,data['y'],color=waterColor,alpha=0.9)
        # Sensor
        if (offset is not None) and (xSensor is not None):
            ax.scatter(xSensor,offset,marker='v',color='k',s=30+scatterSize,zorder=22)
            ax.scatter(xSensor,offset,color='white',s=120+scatterSize+10,edgecolors='k')
            ax.vlines(xSensor, level,offset,linestyles='--',alpha=0.5,color=self.colores_siata[-1])
        #labels
        ax.set_ylabel(yLabel,fontsize=fontsize)
        ax.set_xlabel(xLabel,fontsize=fontsize)
        ax.set_facecolor('white')
        #risks
        if riskLevels is not None:
            x = df['x'].max() -df['x'].min()
            y = df['y'].max() -df['y'].min()
            factorx = 0.05
            ancho = x*factorx
            locx = df['x'].max()+ancho/2.0
            miny = df['y'].min()
            locx = 1.03*locx
            risks = np.diff(np.array(list(riskLevels)+[offset]))
            ax.bar(locx,[riskLevels[0]+abs(miny)],width=ancho,bottom=miny,color='#e3e8f1')
            colors = ['g','orange','red','purple']
            for i,risk in enumerate(risks):
                ax.bar(locx,[risk],width=ancho,bottom=riskLevels[i],color=colors[i],zorder=19,alpha=0.5)
            if level is not None:
                ax.hlines(data['y'].max(),data['x'].max(),locx,lw=1,linestyles='--')
                ax.scatter([locx],[data['y'].max()],s=30,color='k',zorder=20)

    def plot_sectionCurrent(self,*args,**kwargs):
        lastLevel = self.level30Days['Level'].dropna().iloc[-1]/100.0
        self.plot_section(self.lastBat(self.xSensor),figsize = (14,6),offset=self.offset/100.0,level=lastLevel,xSensor=self.xSensor,riskLevels=np.array(self.riskLevels)/100.0,*args,**kwargs)

    def plot_nivel(self,nivel,*args,**kwargs):
        '''Grafica serie de tiempo con valores de profundidad
        |  ----------Parametros
        |  nivel : pd.Series con valores de profundidad
        |  *args : argumentos plt.plot()
        |  **kwargs : riskLevels,resample,riskLevels,xLabel,yLabel,ax,fontsize,figsize,alpha
        |  Nota: todas las unidades en centímetros'''
        # kwargs
        xLabel = kwargs.get('xLabel',False)
        yLabel = kwargs.get('yLabel','Profundidad [cm]')
        fontsize= kwargs.get('fontsize',14)
        figsize = kwargs.get('figsize',(10,4))
        riskLevels = kwargs.get('riskLevels',None)
        ax = kwargs.get('ax',False)
        resample = kwargs.get('resample','5min')
        alpha = kwargs.get('alpha',0.2)
        #plot
        nivel.index = pd.to_datetime(nivel.index)
        if ax == False:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        #1min
        nivel.plot(alpha=0.2,label='',color='r',ax = ax,fontsize=fontsize,rot=False)
        #5min
        if resample <> False:
            nivel = nivel.resample(resample,how='mean')
            nivel.plot(ax=ax,label='',color='k',alpha=1.0,lw=2,fontsize=fontsize)
        #riskLevels
        ax.set_ylabel(yLabel,fontsize=fontsize)
        if riskLevels is not None:
            ax.fill_between(nivel.index,ax.get_ylim()[0],riskLevels[0],alpha=0.1,color=self.colores_siata[0])
            ax.fill_between(nivel.index,riskLevels[0],riskLevels[1],alpha=alpha,color='green')
            ax.fill_between(nivel.index,riskLevels[1],riskLevels[2],alpha=alpha,color='orange')
            ax.fill_between(nivel.index,riskLevels[2],riskLevels[3],alpha=alpha,color='red')
            ax.fill_between(nivel.index,riskLevels[3],max(riskLevels)*1.05,alpha=alpha,color='indigo')
            ax.set_ylim(nivel.min()*0.7,max(riskLevels)*1.05)

    def plot_current(self,*args,**kwargs):
        nivel = self.level30Days.iloc[-180:]['Level']
        fontsize=kwargs.get('fontsize',16)
        figsize = kwargs.get('figsize',(15,11))
        hspace = kwargs.get('hspace',30)
        textPos = kwargs.get('textPos',0.9)
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=None,wspace=None, hspace=hspace)
        #grid
        gs = gridspec.GridSpec(10, 20)
        factor = 6
        ax= plt.subplot(gs[:factor, :])
        ax.set_title(u'Código: %s - Nombre: %s'%(self.codigo,self.infost.loc[self.codigo,'NombreEstacion'].decode('utf-8',errors='replace')),fontsize=18,color = self.colores_siata[-1])
        #Information
        ax2 = plt.subplot(gs[factor:, :9])
        self.plot_nivel(nivel,ax=ax,riskLevels=self.riskLevels)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.text(0.0, -0.15, self.legendText(nivel), color=self.colores_siata[-2],fontsize=fontsize,linespacing=1.6)
        ax2.text(0.0,textPos,'RESUMEN',color = self.colores_siata[-1],fontsize=16,)
        ax2.axis('off')
        #current Section
        ax3 = plt.subplot(gs[factor:, 9:])
        bbox_to_anchor=(1.0,-0.3)
        leg = self.riskLegend(ax3,ncol=4,bbox_to_anchor=bbox_to_anchor,fontsize=fontsize)
        for text in leg.get_texts():
            plt.setp(text, color = self.colores_siata[-2])
        #ax3.set_title(u'Sección transversal y último dato registrado')
        self.plot_sectionCurrent(ax=ax3)

    def plot_nivel30days(self,*args,**kwargs):
        nivel = self.level30Days['Level']
        fontsize=kwargs.get('fontsize',16)
        fig = plt.figure(figsize=(15,13))
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=None,wspace=None, hspace=10)
        #grid
        gs = gridspec.GridSpec(10, 20)
        factor = 6
        ax= plt.subplot(gs[:factor, :])
        ax.set_title(u'Código: %s - Nombre: %s'%(self.codigo,self.infost.loc[self.codigo,'NombreEstacion'].decode('utf-8',errors='replace')),fontsize=18,color = self.colores_siata[-1])
        #Information
        ax2 = plt.subplot(gs[factor:, :9])
        self.plot_nivel(nivel,ax=ax,riskLevels=self.riskLevels)
        ax2.text(0.0, -0.15, self.legendText(nivel), color=self.colores_siata[-2],fontsize=fontsize,linespacing=1.6)
        ax2.text(0.0,0.7,'RESUMEN',color = self.colores_siata[-1],fontsize=16,)
        ax2.axis('off')
        #Section
        ax3 = plt.subplot(gs[factor:, 9:])
        bbox_to_anchor=(1.0,-0.3)
        leg = self.riskLegend(ax3,ncol=4,bbox_to_anchor=bbox_to_anchor,fontsize=fontsize)
        for text in leg.get_texts():
            plt.setp(text, color = self.colores_siata[-2])
        self.plot_section(self.lastBat(self.xSensor),offset=self.offset/100.0,level=nivel.max()/100.0,xSensor=self.xSensor,riskLevels=np.array(self.riskLevels)/100.0,ax=ax3)

    def convert_levelToRisk(self,value,riskLevels):
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
            dif = value - np.array([0]+list(riskLevels))
            return np.argmin(dif[dif >= 0])


    def plot_levelRisk_df(self,df,**kwargs):
        df = df.loc[df.index[::-1]]
        ax = kwargs.get('ax',False)
        c = mcolors.ColorConverter().to_rgb
        cm = self.make_colormap([c('#D9E5E8'),0.20,c('green'),0.4,c('orange'),0.60,c('red'),0.80,c('indigo')])
        if ax == False:
            fig = plt.figure(figsize=(30,30))
            ax = fig.add_subplot(111)
        im = ax.imshow(np.array(df.values,float), interpolation='nearest', vmin=0, vmax=4, aspect='equal',cmap=cm);
        ax.set_xticks(np.arange(0,df.columns.size, 1));
        ax.set_yticks(np.arange(0, df.index.size, 1));
        ax.set_xticklabels(df.columns,fontsize=14,rotation=45,va='center');
        ax.set_yticklabels(df.index,fontsize=14,ha = 'left');
        ax.set_xticks(np.arange(-.5, df.columns.size, 1), minor=True,);
        ax.set_yticks(np.arange(-.5, df.index.size, 1), minor=True);
        plt.draw()
        yax = ax.get_yaxis()
        xax = ax.get_xaxis()
        pad = max(T.label.get_window_extent().width*1.05 for T in yax.majorTicks)
        padx = max(T.label.get_window_extent().width*1.001 for T in xax.majorTicks)
        yax.set_tick_params(pad=pad)
        xax.set_tick_params(pad=padx)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    def risk_dataFrame(self,start,end,convertIndex=True,*args,**kwargs):
        index = pd.to_datetime(pd.date_range(start,end,freq='300s').strftime('%Y-%m-%d %H:%M:00'))
        df = pd.DataFrame(index=index,columns = self.infost[self.infost['estado']=='A'].index)
        for codigo in df.columns:
            try:
                nivel = Nivel(codigo)
                s = nivel.level30Days['Level']
                s.index = pd.to_datetime(s.index)
                s = s.resample('5min',how='mean')
                s = s[~s.index.duplicated(keep='first')]
                s = s.loc[start:end]
                riskLevels = np.copy(nivel.riskLevels)
                df[nivel.codigo] = s.reindex(df.index).apply(lambda x:nivel.convert_levelToRisk(x,riskLevels))
            except:
                pass
        df = df.T
        if convertIndex == True:
            df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]
            df.columns = df.columns.strftime('%H:%M')
            df.index = map(lambda x,y:'%s - %s'%(x,y),df.index,nivel.infost.loc[df.index,'NombreEstacion'].values)
        return df

    def plot_Risk(self,*args):
        self.plot_levelRisk_df(self.risk_dataFrame(*args))

    def plot_estado_riesgo(self):
        end = self.roundTime(pd.to_datetime(datetime.datetime.now()))
        start = end - datetime.timedelta(hours = 3)
        path = '/media/nicolas/Home/Jupyter/MarioLoco/figures/current.png'
        scp = 'scp %s mcano@siata.gov.co:/var/www/mario/realTime/Estado_riesgo_estaciones.png'%path
        self.plot_Risk(start,end)
        plt.savefig(path,bbox_inches='tight')
        os.system(scp)
