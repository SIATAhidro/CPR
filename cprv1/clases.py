
class Nivel(SqlDb):
    '''Class Nivel
    Es una subclase de Main, que hereda sus propiedades'''
    def __init__(self,codigo=None,local={},remote={}):
        SqlDb.__init__(self,codigo=codigo,**remote)

    @property
    def infost(self):
        '''
        Gets full information from all stations
        Returns
        ---------
        pd.DataFrame
        '''
        query = "SELECT * FROM %s"%(self.table)
        return self.read_sql(query).set_index('codigo')

    @property
    def xSensor(self):
        '''Ubicacion del sensor en el eje x'''
        pass
    @property
    def offsetOld(self):
        '''Encuentra el offset de las estaciones de nivel'''
        pass
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
        #return self.read_sqlOld("SELECT action_level, minor_flooding, moderate_flooding, major_flooding FROM estaciones WHERE codigo = '%s'"%self.codigo,toPandas=False)[0]

    @property
    def sensor_type(self):
        '''Encuentra el tipo de sensor,
        ouput:
        1 si el sensor es de tipo radar,
        0 si el sensor es de tipo ultrasonido'''
        #return int(self.read_sqlOld("select N from estaciones where codigo = %s"%self.codigo)[0][0])

    @property
    def get_hbo(self):
        '''pd.DataFrame de historico_bancallena_offset
        hbo = historico_bancallena_offset'''
        df = self.read_sql('select codigo,fecha_hora,offset from historico_bancallena_offset')
        df[df==-9999]=np.NaN
        return df.set_index('codigo')

    def get_level_sensor(self,start,end,calidad=None):
        '''Obtiene datos del sensor a la lámina de agua
        input:
            (start,end) : (fecha inicial,fecha final), format = '%Y-%m-%d %H:%M,
        output:
            pd.Series con los datos del sensor, datos faltantes y con indices malos se llenan con NaN'''
        start = pd.to_datetime(start);end = pd.to_datetime(end)
        datef1 = "DATE_FORMAT(fecha,'%Y-%m-%d')";datef2 = "DATE_FORMAT(hora,'%H:%i:%s')" #formatos de fecha mysql
        # nueva forma de consultar
        query = "SELECT %s,%s,%s FROM datos WHERE cliente = '%s' and calidad = '1' and (((fecha>'%s') or (fecha = '%s' and hora>='%s')) and ((fecha<'%s') or (fecha = '%s' and hora<= '%s')))"%(datef1,datef2,['pr','NI'][self.sensor_type],self.codigo,start.strftime('%Y-%m-%d'),start.strftime('%Y-%m-%d'),start.strftime('%H:%M:00'),end.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),end.strftime('%H:%M:00'))
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

    def read_local_level(self,start,end,offset='dinamico',**kwargs):
        start = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:00')
        end   = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:00')
        sql   = SqlDb(kwargs.get('dbname'),kwargs.get('user'),kwargs.get('table'),
                         kwargs.get('host'),kwargs.get('passwd'),kwargs.get('port'),self.codigo)
        format = (sql.table,sql.codigo,start,end)
        level = sql.read_sql("SELECT fecha,nivel FROM %s WHERE codigo = '%s' and fecha BETWEEN '%s' AND '%s'"%format).set_index('fecha')['nivel']
        if offset == 'dinamico':
            level = self.offset-level
        else:
            level = self.offsetOld-level
        return level


    def create_id_hydro(self):
        '''
        Creates light table with primary keys to make updates faster
        '''
        self.table = 'id_hydro'
        df = self.read_sql("select id,fecha,codigo from hydro")
        S = df[['id','fecha','codigo']].set_index(['codigo','fecha'])
        self.df_to_sql(S.reset_index())
        self.table = 'hydro'

    @property
    def id_df(self):
        return self.read_sql("select fecha,id from id_hydro where codigo = '%s'"%self.codigo).set_index('fecha')['id']



    def update_level_all(self,start,end):
        '''
        Updates sensor level in all stations
        Parameters
        ----------
        start,end    = range to update,datetime or timestamp
        '''
        start_timer = time.time()
        info = Nivel().infost
        codigos = info[info['estado']=='A'].index
        print 'id | time (seconds)  | (%)'
        print '--------------------------'
        for count,codigo in enumerate(codigos):
            start_count = time.time()
            #------
            nivel  = SqlDb(self.dbname,self.user,self.table,self.host,self.passwd,self.port,codigo)
            series = Nivel(nivel.codigo).get_level(start,end)['sensor'].resample('5min').mean()
            nivel.update_series(series,'nivel')
            #-------
            end_count = time.time()
            print '%s | %s | %s'%(codigo,round(time.time()-start_count,2),round(count*100.0/len(codigos),2))
        seconds = end_count-start_timer
        m, s = divmod(seconds, 60)
        print 'Full updating took %s minutes, %s seconds'%(m,s)

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

    def get_areas(self,dfs):
        area = 0
        for df in dfs:
            area+=sum(self.get_area(df['x'].values,df['y'].values))
        return area

    def basin_set_DemDir(self,ruta_dem,ruta_dir,nodata=-9999.0,dxp=12.7):
        wmf.cu.nodata=nodata
        wmf.cu.dxp=dxp
        DEM = wmf.read_map_raster(ruta_dem,True)
        DIR = wmf.read_map_raster(ruta_dir,True)
        DIR[DIR<=0] = wmf.cu.nodata.astype(int)
        DIR = wmf.cu.dir_reclass_rwatershed(DIR,wmf.cu.ncols,wmf.cu.nrows)
        return DEM,DIR

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



class Pluvio(SqlDb):
    def __init__(self,codigo=None,codigos=None):
        SqlDb.__init__(self,codigo)
    @property
    def infost(self):
        infost = self.read_sql('select * from estaciones where red= "pluviografica" or red = "meteorologica" or red = "mocoa-pluvio" and estado="a" and calidad = "1";').set_index('Codigo')
        return infost

    @property
    def sensorRed(self):
        return self.read_sqlOld("select red from estaciones where codigo= '%s'"%self.codigo,toPandas=False)[0][0]

    def read_pluvio(self,start,end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        format_1 = (start.strftime('%Y-%m-%d'),start.strftime('%Y-%m-%d'), start.strftime('%H:%M:00'))
        format_2 = (end.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'),end.strftime('%H:%M:00'))

        query = "SELECT DATE_FORMAT(fecha,'%Y-%m-%d'),DATE_FORMAT(hora,'%H:%i:%s'),P1,P2 "  + \
                "FROM datos "                                                               + \
                "WHERE cliente='%s' and calidad='1' and "              %self.codigo         + \
                "(((fecha>'%s') or (fecha = '%s' and hora>='%s')) and" %format_1            + \
                "((fecha<'%s') or (fecha = '%s' and hora<= '%s')))"    %format_2

        pluvio = self.read_sql(query)
        if pluvio.values.size == 0:
            self.get_data_status = False
            print 'Warning: No data found for station << %s - %d >>, output will be a DataFrame filled with NaN'%(self.nombre,self.codigo)
            df = pd.DataFrame(index=pd.to_datetime(self.datetimeToString(pd.date_range(start,end,freq='min'))),columns=[u'Pluvio'])
            self.data = df

        else:
            if self.sensorRed == 'pluviografica':
                lista = []
                flag=0
                for i,j,k in zip(pluvio.apply(lambda x:'%s %s'%(x[0],x[1][:-3]),axis=1).values,pluvio[pluvio.columns[2]].values,pluvio[pluvio.columns[3]].values):
                    if i == flag:
                        pass
                    else:
                        try:
                            lista.append([pd.to_datetime(i),float(j),float(k)])
                            flag=i
                        except:
                            pass
                df = pd.DataFrame(lista).set_index(0)
                df = df.reindex(pd.to_datetime(self.datetimeToString(pd.date_range(start,end,freq='min'))))
                df.columns = ['P1','P2']
                df = df/1000.0
                df[u'Pluvio']=df.max(axis=1)
            else:
                lista = []
                flag=0
                for i,j in zip(pluvio.apply(lambda x:'%s %s'%(x[0],x[1][:-3]),axis=1).values,pluvio[pluvio.columns[2]].values):
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
                df.columns = ['Pluvio']
                df['Pluvio'] = df['Pluvio']/100.0
        return df['Pluvio']



class Aforos:
    def subsection(self,x,y,v):
        '''Calcula las areas y los caudales de cada
        una de las verticales, con el método de mid-section
        Input:
        x = Distancia desde la banca izquierda, type = numpy array
        y = Produndidad
        v = Velocidad en la vertical
        Output:
        area = Área de la subsección
        Q = Caudal de la subsección
        '''
        # cálculo de áreas
        self.d = np.absolute(np.diff(x))/2.
        self.b = x[:-1]+self.d
        self.area = np.diff(self.b)*y[1:-1]
        self.area = np.insert(self.area, 0, self.d[0]*y[0])
        self.area = np.append(self.area,self.d[-1]*y[-1])
        self.area = np.absolute(self.area)
        # cálculo de caudal
        self.Q = v*self.area
        return self.area,self.Q

    def read_mfpro(self,ruta,**kwargs):
        '''Lee los datos de los archivos generados por el velocímetro mf-pro
        Entrada:
            ruta = ruta donde se encuentra el archivo .
        Salida:
            DataFrame de pandas con los datos observados por el mf-pro
        Nota: la referencia de las  velocidades
        se toma en este caso a partir de la lámina de agua, es decir
        V6 es la velocidad media a 0.6 de la superficie'''
        self.mf = np.loadtxt(ruta,dtype=str, delimiter=None, converters=None, skiprows=32)
        self.dfmf = pd.DataFrame(self.mf)
        self.dfmf = self.dfmf.set_index(0);self.dfmf.index.name='Hora'
        self.dfmf = self.dfmf[[2,5,7,8,9,10,11,12,13,14,15]]
        self.dfmf.columns = ['x','y','Sup','V2','V4','V6','V8','Cama','velocidad04','area','caudal04']
        self.dfmf = self.dfmf.applymap(lambda x: float(string.replace(x, ',', '.')))
        self.cols = ['x','y','velocidad04','area','caudal04']
        self.verticalesmf = range(1,self.dfmf.index.size+1)
        self.verticales = self.dfmf[self.cols]
        self.verticales[self.verticales<0]=0
        self.verticales.index = self.verticalesmf
        self.verticales.index.name = 'vertical'
        self.verticales['caudal04'] = self.subsection(self.verticales['x'].values,\
                              self.verticales['y'].abs().values,
                              self.verticales['velocidad04'].values)[1]
        self.verticales['area']=self.subsection(self.verticales['x'].values,\
                              self.verticales['y'].abs().values,
                              self.verticales['velocidad04'].values)[0]
        return self.verticales

    def plot_section(self,lev,wet,filepath=False,show=False,**kwargs):
        '''Grafica la sección transversal del levantamiento topográfico
        y el aforo.
        self.axis =         kwargs.get('axis',None)
        self.ejex =         kwargs.get('ejex',8)
        self.ejey =         kwargs.get('ejey',4)
        self.paso =         kwargs.get('pasos',2)
        self.poner =        kwargs.get('agregar_indice','no')
        self.quitar =       kwargs.get('quitar_indice','no')
        self.colorscatter = kwargs.get('colorscatter',self.colores_siata[-1])
        self.sizescatter =  kwargs.get('sizescatter',1)
        self.color_lecho =  kwargs.get('color_lecho','tan')
        self.watercolor =   kwargs.get('watercolor',self.colores_siata[0])
        self.fontsize =     kwargs.get('fontsize',14)
        self.ylabel =       kwargs.get('ylabel','Altura [m]')
        self.xlabel =       kwargs.get('xlabel','x [m]')
        self.sepx =         kwargs.get('sepx',0.3)
        self.sepy =         kwargs.get('sepy',0.3)
        self.yticks =       kwargs.get('yticks',1)'''
        #Kwargs
        self.axis =         kwargs.get('axis',None)
        self.ejex =         kwargs.get('ejex',8)
        self.ejey =         kwargs.get('ejey',4)
        self.paso =         kwargs.get('pasos',2)
        self.poner =        kwargs.get('agregar_indice','no')
        self.quitar =       kwargs.get('quitar_indice','no')
        self.colorscatter = kwargs.get('colorscatter',self.colores_siata[-1])
        self.sizescatter =  kwargs.get('sizescatter',1)
        self.color_lecho =  kwargs.get('color_lecho','tan')
        self.watercolor =   kwargs.get('watercolor',self.colores_siata[0])
        self.fontsize =     kwargs.get('fontsize',14)
        self.ylabel =       kwargs.get('ylabel','Altura [m]')
        self.xlabel =       kwargs.get('xlabel','Distancia desde la margen izquierda [m]')
        self.sepx =         kwargs.get('sepx',0.3)
        self.sepy =         kwargs.get('sepy',0.3)
        self.yticks =       kwargs.get('yticks',1)
        self.paso = int(self.paso)
        # Aforo y levantamiento
        self.x = np.array(lev['x'].dropna().values,dtype = float)
        self.y = np.array(lev['y'].dropna().values,dtype = float)
        self.x2 = np.array(wet['x'].values,dtype=float)
        self.y2 = np.array(wet['y'].values,dtype=float)
        figsize = kwargs.get('figsize',(self.ejex,self.ejey))
        #--------DIBUJO DE LA SECCIÓN------------------
        if self.axis == None:
            self.fig=plt.figure(figsize = figsize,edgecolor='r',facecolor='w')
            self.ax=self.fig.add_subplot(111)
        else:
            self.ax = self.axis
        #self.ax.set_aspect(1)
        plt.scatter(self.x,self.y,s=self.sizescatter, zorder = 10, color = self.colorscatter)
        plt.fill_between(self.x2,self.y2,0,color=self.watercolor,zorder = 2)#agua
        plt.fill_between(self.x,self.y,min(self.y),color=self.color_lecho,zorder = 1) #lecho
        plt.ylabel(self.ylabel,fontsize=self.fontsize)
        plt.xlabel(self.xlabel,fontsize=self.fontsize)
        plt.xlim(-self.sepx,max(self.x)+self.sepx)
        if max(self.y) < 0:
            self.ejey = 0
        else:
            self.ejey = max(self.y)
        plt.ylim(min(self.y)-self.sepy,self.ejey+self.sepy)
        self.ax.grid(None)
        if self.yticks == 1:
            plt.yticks([self.y[0],0,min(self.y)],fontsize = self.fontsize)
        elif self.yticks == 2:
            plt.yticks([0,min(self.y)],fontsize = self.fontsize)
        else:
            plt.yticks([self.y[0],min(self.y)],fontsize = self.fontsize)
        self.xticks = self.x2[::self.paso]
        if self.poner <> 'no':
            self.poner = int(self.poner)
            self.xticks = np.append(self.xticks,self.x[self.poner])
        plt.xticks(self.xticks,fontsize = self.fontsize)
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath,format=filepath[-3:],bbox_inches='tight')
        if show == True:
            plt.show()

    def resultado_mfpro(self,ruta,vel=False,**kwargs):
        self.verticales = self.read_mfpro(ruta)
        # Fecha
        with open(ruta, 'r') as f:
            for i in range(3):
                self.read_data = f.readline()
        self.date = '%s-%s-%s %s'%(self.read_data[16:20],self.read_data[13:15],self.read_data[10:12],self.read_data[:5])
        self.fecha = pd.to_datetime(self.date)
        if vel == True:
            self.vel_csv = pd.read_csv('%s/%s/%s_%s.csv'%(self.ruta,self.nombre,self.nombre,self.aforo),skiprows=4,index_col=0)['velocidad08']
            self.verticales['velocidad08'] = self.vel_csv
            self.verticales['caudal08'] = 0.8*self.subsection(self.verticales['x'].values,self.verticales['y'].values,self.verticales['velocidad08'])[1]
            self.verticales = self.verticales[['x','y','velocidad04','velocidad08','area','caudal04','caudal08']]
        else:
            self.verticales['velocidad08'] = np.nan
            self.verticales['caudal08'] = np.nan
        #guarda verticales del aforo
        self.df_aforo=self.verticales
        self.x2 = self.verticales['x'].values
        self.y2 = self.verticales['y'].values
        self.p = []
        for i in range(len(self.x2)-1):
            self.p.append(float(np.sqrt(abs(self.x2[i]-self.x2[i+1])**2.0+abs(self.y2[i]-self.y2[i+1])**2.0)))
        #-------------------------RESULTADOS AFORO---------------------------------------
        self.indices = ['id_estacion_asociada','dispositivo','fecha','ancho_superficial',
                 'caudal_medio','caudal_superficial','error_caudal','velocidad_media',\
                 'velocidad_superficial','perimetro','area_total','altura_media',\
                 'radio_hidraulico','flag_izquierda','flag_ubicacion','source','descripcion']
        self.dispositivo = 'MF-PRO'
        self.flag_ubicacion = kwargs.get('flag_ubicacion',1)
        self.source = kwargs.get('source','siata')
        self.valores = [self.codigo,self.dispositivo,self.fecha,
                  max(self.x2)- min(self.x2),self.df_aforo['caudal04'].sum(),\
                  self.df_aforo['caudal08'].sum(),-999,self.df_aforo['caudal04'].sum()/self.df_aforo['area'].sum(),\
                  self.df_aforo['caudal08'].sum()/self.df_aforo['area'].sum(),sum(self.p),self.df_aforo['area'].sum(),\
                  abs(np.mean(self.y2)),self.df_aforo['area'].sum()/sum(self.p),1,self.flag_ubicacion,self.source,'']
        self.dfr = pd.DataFrame(self.valores,index = self.indices,columns=['Resultado'])
        self.dfr.index.name = 'Parametro'
        savedata = kwargs.get('savedata',False)
        if savedata == True:
            self.verticales.to_csv('%s/%s/%s_aforo_%s.csv'%(self.ruta_salida,self.nombre,self.codigo,self.aforo))
            self.dfr.to_csv('%s/%s/%s_resultado_%s.csv'%(self.ruta_salida,self.nombre,self.codigo,self.aforo))
        self.verticales['y'] = self.verticales['y']*-1.


    def aforo_latex(self,**kwargs):
        '''Genera una sección de latex con todos los resultados de la sección
        este archivo, se utiliza luego con \include{...} en cualquier informe
        SALIDA:
        archivo de latex .tex, contiene descripción del sitio de aforo,
        foto del sitio de aforo, mapa, información de la estación de nivel más
        cercana, instrumento utilizado en el aforo, gráfica del levantamiento,
        resultados de los parámetros hidrálicos, gráfica de caudal y velocidad
        en las verticales elejidas durante el aforo.
        kwargs:
            redrio: False - si es para el informe redrío
            name: nombre que aparecerá en el informe

        '''
        self.redrio = kwargs.get('redrio',False)
        #RESULTADOS DEL AFORO
        self.alo = []
        #if self.aforo <> 1:
        self.alo.append('\\null\\newpage')
        self.alo.append('\\subsection{Resultados del aforo}')
        self.alo.append('\\begin{table}[h!]')
        self.alo.append('\\caption{Resultados %s, Aforo %s}'%(self.name,self.date))
        self.alo.append('\\vspace{1.9mm}')
        self.alo.append('\\fontsize{10}{2} \\selectfont{')
        self.parametros = '\\scalebox{1.0}{\\begin{tabularx}{\\textwidth}{ p{1.4in} p{1.3in} p{1.7in} p{1.3in}}'
        self.alo.append(self.parametros)
        self.alo.append('\\rowcolor{CiceBlue2}')
        self.alo.append('\\multicolumn{4}{c}{\\textbf{\\textcolor{white}{RESULTADOS DEL AFORO}}}')
        self.alo.append('\\rule[-0.2cm]{0cm}{0.7cm}\\\\')
        self.campo = ['Caudal [$ m^{3} $]','Caudal S [$ m^{3}/s $]','Área [$ m^{2}$]',\
                'Ancho Superficial [$ m $]','Altura Media [$ m $] ' ,'Velocidad Media [$ m/s $]',\
                'Perímetro M [$ m $]','Radio Hidráulico [$ m $]','Offset [$ m $]','Med-Nivel(UTH) [$ m $]']
        self.caudal = float(self.dfr.loc['caudal_medio','Resultado'])
        self.caudals = float(self.dfr.loc['caudal_superficial','Resultado'])
        self.area = float(self.dfr.loc['area_total','Resultado'])
        self.ancho = float(self.dfr.loc['ancho_superficial','Resultado'])
        self.altura = float(self.dfr.loc['altura_media','Resultado'])
        self.velm = float(self.dfr.loc['velocidad_media','Resultado'])
        self.perimetro =float(self.dfr.loc['perimetro','Resultado'])
        self.radio = float(self.dfr.loc['radio_hidraulico','Resultado'])
        # PARA LEER ARCHIVOS SIATA
        try:
            self.qs = query_siata(self.fecha+datetime.timedelta(hours = -3),self.fecha, self.codigo)
            self.offset = self.qs.get_offset()/100.
            self.niveluth = self.qs.get_level().mean()
        except:
            self.offset = -999
            self.niveluth = -999
        if redrio == False:
            self.alo.append('\\textbf{%s}'%self.campo[0])
            self.alo.append('& %.3f'%self.caudal)
            self.alo.append('& \\textbf{%s}'%self.campo[1])
            self.alo.append('& %.3f'%self.caudals)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[2])
            self.alo.append('& %.3f'%self.area)
            self.alo.append('& \\textbf{%s}'%self.campo[3])
            self.alo.append('& %.3f'%self.ancho)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[4])
            self.alo.append('& %.3f'%self.altura)
            self.alo.append('& \\textbf{%s}'%self.campo[5])
            self.alo.append('& %.3f'%self.velm)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[6])
            self.alo.append('& %.3f'%self.perimetro)
            self.alo.append('& \\textbf{%s}'%self.campo[7])
            self.alo.append('& %.3f'%self.radio)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[8])
            self.alo.append('& %.3f'%self.offset)
            self.alo.append('& \\textbf{%s}'%self.campo[9])
            self.alo.append('& %.3f'%self.niveluth)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\label{tab:resultados%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{tabularx}}}')
            self.alo.append('\\end{table}')
            # GRAFICA DEL AFORO
            self.ruta_plot = '%s/%s/grafica_aforo_%d.pdf'%(self.ruta_salida,self.nombre,self.aforo)
            self.alo.append('\\begin{figure}[h!]')
            self.alo.append('\\centering')
            self.plot_height = kwargs.get('plot_height',3.0)
            self.size_plot = kwargs.get('size','height=%scm'%self.plot_height)
            self.alo.append('\\includegraphics[%s]{%s}'%(self.size_plot,self.ruta_plot))
            self.alo.append('\\caption{Dibujo de la sección}')
            self.alo.append('\\label{fig:dibujo%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{figure}')
            # GRÁFICA DE VERTICALES
            self.alo.append('\\begin{figure}[h!]')
            self.alo.append('\\centering')
            self.ruta_plot = '%s/%s/verticales_%d.pdf'%(self.ruta_salida,self.nombre,self.aforo)
            self.alo.append('\\includegraphics[width=12.0cm]{%s}'%self.ruta_plot)
            self.alo.append('\\caption{Resultados por verticales}')
            self.alo.append('\\label{fig:verticales%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{figure}')
            self.clearpage = kwargs.get('clearpage',True)
            if self.clearpage == True:
                self.alo.append('\\clearpage')
            self.alop=open('%s/%s/resultados_%s_%s.tex'%(self.ruta,self.nombre,self.nombre,self.aforo), 'w')
            np.savetxt('%s/%s/resultados_%s_%s.tex'%(self.ruta,self.nombre,self.nombre,self.aforo), self.alo, fmt='%s')
            self.alop.close()
        else:
            self.alo.append('\\textbf{%s}'%self.campo[0])
            self.alo.append('& %.3f'%self.caudal)
            self.alo.append('& \\textbf{Dispositivo}')
            self.alo.append('& %s'%self.dispositivo)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[2])
            self.alo.append('& %.3f'%self.area)
            self.alo.append('& \\textbf{%s}'%self.campo[3])
            self.alo.append('& %.3f'%self.ancho)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[4])
            self.alo.append('& %.3f'%self.altura)
            self.alo.append('& \\textbf{%s}'%self.campo[5])
            self.alo.append('& %.3f'%self.velm)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\textbf{%s}'%self.campo[6])
            self.alo.append('& %.3f'%self.perimetro)
            self.alo.append('& \\textbf{%s}'%self.campo[7])
            self.alo.append('& %.3f'%self.radio)
            self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
            self.alo.append('\\label{tab:resultados%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{tabularx}}}')
            self.alo.append('\\end{table}')
            # GRAFICA DEL AFORO
            self.ruta_plot = '%s/%s/grafica_aforo_%d.pdf'%(self.ruta_salida,self.nombre,self.aforo)
            self.alo.append('\\begin{figure}[h!]')
            self.alo.append('\\centering')
            self.plot_height = kwargs.get('plot_height',5.0)
            self.size_plot = kwargs.get('size','height=%scm'%self.plot_height)
            self.alo.append('\\includegraphics[%s]{%s}'%(self.size_plot,self.ruta_plot))
            self.alo.append('\\caption{Dibujo de la sección}')
            self.alo.append('\\label{fig:dibujo%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{figure}')
            # GRÁFICA DE VERTICALES
            self.alo.append('\\begin{figure}[h!]')
            self.alo.append('\\centering')
            self.ruta_plot = '%s/%s/verticales_%d.pdf'%(self.ruta_salida,self.nombre,self.aforo)
            self.alo.append('\\includegraphics[width=12.0cm]{%s}'%self.ruta_plot)
            self.alo.append('\\caption{Resultados por verticales}')
            self.alo.append('\\label{fig:verticales%s%s}'%(self.codigo,self.aforo))
            self.alo.append('\\end{figure}')
            self.clearpage = kwargs.get('clearpage',True)
            if self.clearpage == True:
                self.alo.append('\\clearpage')
            self.alop=open('%s/%s/resultados_%s_%s.tex'%(self.ruta,self.nombre,self.nombre,self.aforo), 'w')
            np.savetxt('%s/%s/resultados_%s_%s.tex'%(self.ruta,self.nombre,self.nombre,self.aforo), self.alo, fmt='%s')
            self.alop.close()

    def includegraphics(self,include,name,caption,label,width=16.0,multicols=False):
        '''Función para facilitar la creación de figuras, labels y captions'''
        if multicols==True:
            include.append('\\lipsum[1]')
            include.append('\\begin{Figure}')
            include.append(' \\centering')
            include.append(' \\includegraphics[width=%scm]{%s}'%(width,name))
            include.append(' \\captionof{figure}{%s}'%caption)
            include.append('\\end{Figure}')
        else:
            include.append('\\begin{figure}[h!]')
            include.append('\\begin{center}')
            include.append('\\includegraphics[width=%scm]{%s}'%(width,name))
            include.append('\\vspace{+3mm}')
            include.append('\\caption{\small %s}'%caption)
            include.append('\\vspace{+5mm}')
            include.append('\\label{%s}'%label)
            include.append('\\end{center}')
            include.append('\\end{figure}')

    def subir_ott_an(self):
        '''Inserta tabla de resultados en aforo_nueva'''
        self.ott_id_aforo()
        # Insertar resultados
        if self.id_aforo==-999:
            self.insert = 'INSERT INTO aforo_nueva (%s) VALUES (%s)'%(str(list(self.dfr.index)).strip('[]').replace("'",""),
                                                             str(list(self.dfr['Resultado'].values)).strip('[]') )
            self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
            self.db_cursor = self.conn_db.cursor ()
            try:
                self.db_cursor.execute(self.insert)
                self.conn_db.commit()
                self.conn_db.close ()
                self.ott_id_aforo()
                print 'INSERT INTO in aforo_nueva SUCCEED'
                self.subir_ott_an_status = 'worked'
            except:
                self.conn_db.close ()
                print 'INSERT INTO in aforo_nueva did not work'
                self.ott_id_aforo()
                pass
        else:
            print 'El aforo ya se encuentra en la base de datos'

    def subir_an(self):
        '''Inserta tabla de resultados en aforo_nueva'''
        self.get_id_aforo()
        self.set_mysql()
        if  self.id_aforo==-999:
	    print 'El aforo no se encuentra en la base de datos'
	    self.insert = 'INSERT INTO aforo_nueva (%s) VALUES (%s)'%(str(list(self.dfr.index)).strip('[]').replace("'",""),str(list(self.dfr['Resultado'].fillna(-999).values)).strip('[]') )
	    self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
	    print 'connection stablished'
	    self.db_cursor = self.conn_db.cursor ()
	    print 'database connection stablished'
	    print 'print connection closed'
	    try:
		self.db_cursor.execute(self.insert)
                print 'insert executed'
                self.conn_db.commit()
                self.subir_an_status = 'subir_an worked'
                print 'commit done'
            except:
                print 'INSERT INTO in aforo_nueva did not work'
                self.subir_an_status = 'subir_an failed'
                pass
            self.conn_db.close()
        else:
            print 'El aforo ya se encuentra en la base de datos'
	    self.subir_an_status = 'already in database'

    def subir_ott_san(self):
        '''Inserta tabla en seccion_aforo_nueva'''
        self.ott_id_aforo()
        self.verticales = self.verticales.fillna(-999)
        if self.id_aforo <> -999:
            # Insertar en seccion_aforo_nueva
            for i in range(self.verticales.index.size):
                self.indices = 'id_aforo, vertical, %s'%str(list(self.verticales.columns)).strip('[]').replace("'","")
                self.vertical = self.verticales.index[i]
                if self.vertical in self.validate('seccion_aforo_nueva'):
                    print 'la vertical %s en id_aforo %s ya se encuentra en la tabla seccion_aforo_nueva'%(self.vertical,self.id_aforo)
                else:
                    self.lista = list(self.verticales.iloc[i].values)
                    self.lista.insert(0,self.vertical)
                    self.lista.insert(0,self.id_aforo)
                    self.valores = str(map(lambda x:str(x),self.lista)).strip('[]')
                    self.insert = "INSERT INTO seccion_aforo_nueva (%s) VALUES (%s)"%(self.indices,self.valores)
                    self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
                    try:
						self.db_cursor = self.conn_db.cursor()
						self.db_cursor.execute (self.insert)
						self.conn_db.commit()
						self.conn_db.close()
						print 'Insert in %s, vertical = %s inserted'%(self.id_aforo,self.vertical)
                    except:
						print 'INSERT INTO in seccion_aforo_nueva vertical %s did not work'%self.vertical
						self.conn_db.close()
						pass

    def subir_san(self):
        '''Inserta tabla en seccion_aforo_nueva'''
        self.get_id_aforo()
        self.set_mysql()
        if self.id_aforo <> -999:
            # Insertar en seccion_aforo_nueva
            for i in range(self.verticales.index.size):
                self.indices = 'id_aforo, vertical, %s'%str(list(self.verticales.columns)).strip('[]').replace("'","")
                self.vertical = self.verticales.index[i]
                if self.vertical in self.validate('seccion_aforo_nueva'):
                    print 'la vertical %s en id_aforo %s ya se encuentra en la tabla seccion_aforo_nueva'%(self.vertical,self.id_aforo)
                else:
                    self.lista = list(self.verticales.iloc[i].values)
                    self.lista.insert(0,self.vertical)
                    self.lista.insert(0,self.id_aforo)
                    self.valores = str(map(lambda x:str(x),self.lista)).strip('[]')
                    self.insert = "INSERT INTO seccion_aforo_nueva (%s) VALUES (%s)"%(self.indices,self.valores)
                    self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
                    try:
                        self.db_cursor = self.conn_db.cursor()
                        self.db_cursor.execute (self.insert)
                        self.conn_db.commit()
                        self.conn_db.close()
                        print 'vertical = %s insert Worked'%self.vertical
                    except:
                        print 'vertical %s insert Failed'%self.vertical
                        self.conn_db.close()
                        pass

    def subir_ott_lan(self):
        '''Inserta tabla en levantamiento_aforo_nueva'''
        self.ott_id_aforo()
        if self.id_aforo <> -999:
            # Insertar en levantamiento_aforo_nueva
            for i in range(self.levantamiento.index.size):
                self.vertical = self.levantamiento.index[i]
                if self.vertical in self.validate('levantamiento_aforo_nueva'):
                        print 'la vertical %s en id_aforo %s ya se encuentra en la tabla %s'%(self.vertical,self.id_aforo,'levantamiento_aforo_nueva')
                else:
                    self.xl = self.levantamiento.iloc[i][0]
                    self.yl = self.levantamiento.iloc[i][1]
                    self.valores = "'%s' ,'%s', %s, %s"%(self.id_aforo,self.vertical,self.xl,self.yl)
                    self.insert = "INSERT INTO levantamiento_aforo_nueva (id_aforo, vertical, x, y) VALUES (%s)"%(self.valores)
                    self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
                    try:
                        self.db_cursor = self.conn_db.cursor()
                        self.db_cursor.execute (self.insert)
                        self.conn_db.commit ()
                        self.conn_db.close()
                        print 'Insert in %s, vertical = %s inserted'%(self.id_aforo,self.vertical)
                    except:
                        print 'INSERT INTO in levantamiento_aforo_nueva vertical %s did not work'%self.vertical
                        self.conn_db.close()
                        pass

    def subir_lan(self):
        '''Inserta tabla en levantamiento_aforo_nueva'''
        if self.id_aforo <> -999:
            # Insertar en levantamiento_aforo_nueva
            for i in range(self.levantamiento.index.size):
                self.vertical = self.levantamiento.index[i]
                if self.vertical in self.validate('levantamiento_aforo_nueva'):
                        print 'la vertical %s en id_aforo %s ya se encuentra en la tabla %s'%(self.vertical,self.id_aforo,'levantamiento_aforo_nueva')
                else:
                    self.xl = self.levantamiento.iloc[i][0]
                    self.yl = self.levantamiento.iloc[i][1]
                    self.valores = "'%s' ,'%s', %s, %s"%(self.id_aforo,self.vertical,self.xl,self.yl)
                    self.insert = "INSERT INTO levantamiento_aforo_nueva (id_aforo, vertical, x, y) VALUES (%s)"%(self.valores)
                    self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
                    try:
                        self.db_cursor = self.conn_db.cursor()
                        self.db_cursor.execute (self.insert)
                        self.conn_db.commit ()
                        self.conn_db.close()
                        print 'Insert in %s, vertical = %s inserted'%(self.id_aforo,self.vertical)
                    except:
                        print 'INSERT INTO in levantamiento_aforo_nueva vertical %s did not work'%self.vertical
                        self.conn_db.close()
                        pass
    def resultados(self,dispositivo,margen,flag_ubicacion,vel,**kwargs):
        self.dispositivo = dispositivo
        if vel == True:
            pass
        else:
            self.verticales['velocidad08'] = np.nan
            self.verticales['caudal08'] = np.nan
        #guarda verticales del aforo
        self.x2 = self.verticales['x'].values
        self.y2 = -1*self.verticales['y'].values
        self.area_i,self.caudal_i = self.subsection(self.x2,self.y2,self.verticales['velocidad04'].abs().values)
        self.caudals_i= 0.8*self.subsection(self.x2,self.y2,self.verticales['velocidad08'].abs().values)[1]
        self.verticales['area'] = self.area_i
        self.verticales['caudal04'] =  self.caudal_i; self.verticales['caudal08'] =  self.caudals_i
        self.p = []
        if self.y2[0]<0.0:
            self.p.append(abs(self.x2[0]))
        if self.y2[-1]<0.0:
            self.p.append(abs(self.x2[-1]))
        for i in range(len(self.x2)-1):
            self.p.append(float(np.sqrt(abs(self.x2[i]-self.x2[i+1])**2.0+abs(self.y2[i]-self.y2[i+1])**2.0)))
        #-------------------------RESULTADOS AFORO---------------------------------------
        self.indices = ['id_estacion_asociada','dispositivo','fecha','ancho_superficial',
                 'caudal_medio','caudal_superficial','error_caudal','velocidad_media',\
                 'velocidad_superficial','perimetro','area_total','altura_media',\
                 'radio_hidraulico','flag_izquierda','flag_ubicacion','source','descripcion']

        self.valores = [self.codigo,self.dispositivo,self.fecha,max(self.x2)- min(self.x2),
                        self.verticales['caudal04'].sum(), self.verticales['caudal08'].sum(),\
                        -999,self.verticales['caudal04'].sum()/self.verticales['area'].sum(),\
                        self.verticales['caudal08'].sum()/self.verticales['area'].sum(),\
                        sum(self.p),self.verticales['area'].sum(), self.verticales['area'].sum()/(max(self.x2)- min(self.x2)),\
                        self.verticales['area'].sum()/sum(self.p),margen,flag_ubicacion,self.source,'']

        self.dfr = pd.DataFrame(self.valores,index = self.indices,columns=['Resultado']).fillna(-999)
        self.dfr.index.name = 'Parametro'
    def update_an(self):
        '''Actualiza tabla de resultados en aforo_nueva'''
        # Insertar resultado
        self.dfr = self.dfr.fillna(-999)
        string = ''
        for i,j in zip(self.dfr.index,self.dfr['Resultado'].values):
            if i <> self.dfr.index[-1]:
                string = string + "%s = '%s', "%(i,j)
            else:
                string = string + "%s = '%s'"%(i,j)
        self.insert = "UPDATE aforo_nueva SET %s WHERE id_aforo = '%s';"%(string,self.id_aforo)
        self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
        self.db_cursor = self.conn_db.cursor ()
        try:
            self.db_cursor.execute(self.insert)
            self.conn_db.commit()
            self.conn_db.close ()
            print 'UPDATE in aforo_nueva SUCCEED'
            self.update_status = 'worked'
        except:
            self.conn_db.close ()
            print 'UPDATE in aforo_nueva did not work'
            pass

    def execute_mysql(self,execution):
        print 'EXECUTION:\n %s'%execution
        self.conn_db = MySQLdb.connect (self.host, self.user, self.passwd, self.dbname)
        self.db_cursor = self.conn_db.cursor ()
        try:
            self.db_cursor.execute(execution)
            self.conn_db.commit()
            self.conn_db.close ()
            status = 'worked'
        except:
            self.conn_db.close ()
            status = 'failed'
            pass
        print 'EXECUTION STATUS: %s \n'%status

    def lista_update(self,dbase,show_print=True):
        show_print = True
        if dbase =='seccion_aforo_nueva':
            self.subir_san()
            id_dbase = 'id_seccion_aforo'
            dfloc = self.verticales
        else:
            self.subir_lan()
            id_dbase = 'id_levantamiento'
            dfloc = self.levantamiento
        # dataframe database
        dfdb = self.get_aforo_mysql(dbase).set_index('vertical')[[id_dbase]+list(dfloc.columns)]
        # dataframe local
        dfloc = dfloc.fillna(-999)
        to_remove = list((set(list(dfdb.index)) - set(list(dfloc.index))))
        executions = []
        for i,j in zip(list(dfdb.index),dfdb[id_dbase].values):
                execution = ''
                id_dbase_value = j
                if i in to_remove:
                    executions.append("DELETE from %s where %s = %s"%(dbase,id_dbase,id_dbase_value))
                else:
                    for j,k in enumerate(dfdb.loc[i].index[1:]):
                        value = dfloc.loc[i].values[j]
                        vertical = i
                        if k == dfdb.loc[i].index[-1]:
                            execution = execution + "%s = '%s' "%(k,value)
                        else:
                            execution = execution + "%s = '%s', "%(k,value)
                    execution = "UPDATE %s SET vertical = %s, %s WHERE %s = '%s';"%(dbase,vertical,execution,id_dbase,id_dbase_value)
                executions.append(execution)
                if show_print ==True:
                    print i
                    print execution
                    print ''
        return executions

    def update(self,dbase):
        self.get_id_aforo()
        if dbase=='aforo_nueva':
            self.update_an()
        else:
            executions = self.lista_update(dbase,show_print=False)
            for execution in executions:
                self.execute_mysql(execution)
    def data_to_excel(self,filepath,showprint=False,**kwargs):
        ''' converts aforos information into an excel file  Parameters
        ----------
        filepath: path to store excel file
        showprint: print outcome
        -------
        kwags :
        -------
        Returns
            ------
            workbook: excelfile stored in filepath'''
        self.sensor_info = [-999]*6
        dfr = self.dfr.fillna(-999)
        #sensor = ['n1','n2','n3','n4','offset','x_sensor']
        verticales_width = 20
        levantamiento_width = 15
        self.dfr.loc['descripcion','Resultado'] = -999
        # EXCELFILE
        workbook = xlsxwriter.Workbook(filepath)
        # Formats
        format1 = workbook.add_format({
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#79A2CE',
            'font_color': 'white'})
        format2 = workbook.add_format({
            'border': 1,
            'align':'center',
            'valign': 'vcenter',
            'fg_color': 'white'})
        format4 = workbook.add_format({
            'border': 1,
            'align':'center',
            'valign': 'vcenter',
            'fg_color': 'white',
            'num_format': 'yyyy-mm-dd hh:mm',
            })
        format3 = workbook.add_format({
            'bold': 1,
            'border': 1,
            'valign': 'vcenter'})
        # SHEETS
        # Sheet: Informacion
        #columns width
        informacion = workbook.add_worksheet(name='Informacion')
        informacion.set_column(0,0,30)
        informacion.set_column(1,1,25)
        # columns names
        informacion.write(0,0,'Nombre', format1)
        informacion.write(0,1,'Valor', format1)
        informacion.write(0,2,'Unidad', format1)
        # values
        params = ['Nombre','Municipio',u'Dirección','Barrio','Subcuenca','Longitud',
         'Latitud','Minor flooding','Moderate flooding','Major flooding','Action level','Offset','X sensor']
        values = [self.name,self.municipio,self.direccion,self.barrio,self.subcuenca,
                  self.longitud,self.latitud]+self.sensor_info
        values = map(lambda x: codecs.utf_8_decode(str(x))[0],values)
        unidades = ['','','','','','','','m','m','m','m','m']
        for j,valor,resultado,unidad in zip(range(1,1+len(params)),params,values,unidades):
            informacion.write(j,0,valor,format3)
            informacion.write(j,1,resultado,format2)
            informacion.write(j,2,unidad,format2)
        # Sheet Resultados
        resultados = workbook.add_worksheet(name='Resultados')
        #columns width
        resultados.set_column(0,0,30)
        resultados.set_column(1,1,30)
        resultados.set_column(2,2,20)
        # columns names
        resultados.write(0,0,'Nombre', format1)
        resultados.write(0,1,'Valor', format1)
        resultados.write(0,2,'Unidad', format1)
        # values
        unidades = [' ',' ',' ','m','m^3/s','m^3/s','m^3/s','m/s','m/s','m','m^2','m','m',' ',' ',' ',' ',]
        for i,valor,resultado,unidad in zip(range(1,self.dfr.index.size+1),self.dfr.index.values,self.dfr['Resultado'].values,unidades):
            resultados.write(i,0,valor,format3)
            resultados.write(i,1,resultado,format2)
            resultados.write(i,2,unidad,format2)
        resultados.write(4,1,self.fecha,format4)
        # Sheet: Verticales
        verticales = workbook.add_worksheet(name='Verticales')
        verticales.set_column(0,0,10)
        # columns names
        columns = ['Vertical','X [m]','Y [m]','Velocidad04 [m/s]','Velocidad08 [m/s]','Caudal08 [m/s]','Area [m^2]','Caudal04 [m^3/s]']
        for col in range(1,9):
            verticales.set_column(col,col,verticales_width)
            verticales.write(0,col-1,columns[col-1], format1)
        # values
        self.verticales = self.verticales.fillna(-999)
        for vertical in self.verticales.index.values:
            verticales.write(vertical,0,vertical,format1)
            for idcol,colname in zip(range(1,self.verticales.columns.size+1),self.verticales.columns):
                verticales.write(vertical,idcol,self.verticales.loc[vertical,colname],format2)
        # Sheet: Levantamiento
        levantamiento = workbook.add_worksheet(name='Batimetria')
        levantamiento.set_column(0,0,10)
        # columns names
        columns = ['Vertical','X [m]','Y [m]']
        for col in [1,2,3]:
            levantamiento.set_column(col,col,levantamiento_width)
            levantamiento.write(0,col-1,columns[col-1], format1)
        # values
        self.levantamiento = self.levantamiento.fillna(-999)
        for vertical in self.levantamiento.index.values:
            levantamiento.write(vertical,0,vertical,format1)
            for idcol,colname in zip(range(1,self.levantamiento.columns.size+1),self.levantamiento.columns):
                levantamiento.write(vertical,idcol,self.levantamiento.loc[vertical,colname],format2)
        workbook.close()
        showprint = kwargs.get('showprint',False)
        if showprint:
            print 'Excel file stored in: %s'%(filepath)

    def latex_results_table(self,label):
        self.alo = ['','\\ % ----------------------latex_results_table ---------------------------------------']
        #if self.aforo <> 1:
        self.alo.append('\\null\\newpage')
        self.alo.append('\\subsection{%s}'%self.name)
        self.alo.append('\\begin{table}[h!]')
        self.alo.append('\\caption{Resultados %s, Aforo %s}'%(self.name,self.date))
        self.alo.append('\\vspace{1.9mm}')
        self.alo.append('\\fontsize{10}{2} \\selectfont{')
        self.parametros = '\\scalebox{1.0}{\\begin{tabularx}{\\textwidth}{ p{1.4in} p{1.3in} p{1.7in} p{1.3in}}'
        self.alo.append(self.parametros)
        self.alo.append('\\rowcolor{CiceBlue2}')
        self.alo.append('\\multicolumn{4}{c}{\\textbf{\\textcolor{white}{RESULTADOS DEL AFORO}}}')
        self.alo.append('\\rule[-0.2cm]{0cm}{0.7cm}\\\\')
        self.campo = ['Caudal [$ m^{3} $]','Caudal S [$ m^{3}/s $]','Área [$ m^{2}$]',\
                'Ancho Superficial [$ m $]','Altura Media [$ m $] ' ,'Velocidad Media [$ m/s $]',\
                'Perímetro M [$ m $]','Radio Hidráulico [$ m $]','Offset [$ m $]','Med-Nivel(UTH) [$ m $]']
        self.caudal = float(self.dfr.loc['caudal_medio','Resultado'])
        self.caudals = float(self.dfr.loc['caudal_superficial','Resultado'])
        self.area = float(self.dfr.loc['area_total','Resultado'])
        self.ancho = float(self.dfr.loc['ancho_superficial','Resultado'])
        self.altura = float(self.dfr.loc['altura_media','Resultado'])
        self.velm = float(self.dfr.loc['velocidad_media','Resultado'])
        self.perimetro =float(self.dfr.loc['perimetro','Resultado'])
        self.radio = float(self.dfr.loc['radio_hidraulico','Resultado'])
        self.alo.append('\\textbf{%s}'%self.campo[0])
        self.alo.append('& %.3f'%self.caudal)
        self.alo.append('& \\textbf{Dispositivo}')
        self.alo.append('& %s'%self.dispositivo)
        self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
        self.alo.append('\\textbf{%s}'%self.campo[2])
        self.alo.append('& %.3f'%self.area)
        self.alo.append('& \\textbf{%s}'%self.campo[3])
        self.alo.append('& %.3f'%self.ancho)
        self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
        self.alo.append('\\textbf{%s}'%self.campo[4])
        self.alo.append('& %.3f'%self.altura)
        self.alo.append('& \\textbf{%s}'%self.campo[5])
        self.alo.append('& %.3f'%self.velm)
        self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
        self.alo.append('\\textbf{%s}'%self.campo[6])
        self.alo.append('& %.3f'%self.perimetro)
        self.alo.append('& \\textbf{%s}'%self.campo[7])
        self.alo.append('& %.3f'%self.radio)
        self.alo.append('\\rule[-0.1cm]{0cm}{0.5cm} \\\\ \\cline{1-4}')
        self.alo.append('\\label{%s}'%(label))
        self.alo.append('\\end{tabularx}}}')
        self.alo.append('\\end{table}')
        self.alo.append('\\ %----------------------LABEL------------------')
        self.alo.append('\\ % -----Tabla '+'\\ref{%s}'%label)
        return self.alo

class Plots:
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



    def set_cu(self):
        self.cu = wmf.SimuBasin(rute='/media/nicolas/maso/Mario/basins/%s.nc'%self.codigo)

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


    def set_environment(sekf):
        # SETTING PLOTS ENVIRONMENT
        plt.style.use('seaborn-dark')
        plt.rc('font', family=fm.FontProperties(fname='/media/nicolas/Home/Jupyter/MarioLoco/Tools/AvenirLTStd-Book.ttf',).get_name())
        typColor = '#%02x%02x%02x' % (8,31,45)
        plt.rc('axes',labelcolor=typColor)
        plt.rc('axes',edgecolor=typColor)
        plt.rc('text',color= typColor)
        plt.rc('xtick',color=typColor)
        plt.rc('ytick',color=typColor)

print 'worked'
