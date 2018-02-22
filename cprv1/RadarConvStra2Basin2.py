#!/usr/bin/env python

from wmf import wmf 
#import func_SIATA as fs
import netCDF4
import pylab as pl
import numpy as np
import datetime as dt
import argparse
import textwrap
import os 
import pandas as pd
import glob

#Parametros de entrada del trazador
parser=argparse.ArgumentParser(
	prog='RadarStraConv2Basin',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	description=textwrap.dedent('''\
	Toma los campos de precip, conv y stratiformes tipo nc y los 
        convierte al formato de la cuenca, esta segunda version
		obtiene tambien los campos con intervalos maximos y minimos
		de precipitacion.
        '''))
#Parametros obligatorios
parser.add_argument("fechaI",help="(YYYY-MM-DD) Fecha de inicio de imagenes")
parser.add_argument("fechaF",help="(YYYY-MM-DD) Fecha de fin de imagenes")
parser.add_argument("cuenca",help="(Obligatorio) Ruta de la cuenca en formato .nc")
parser.add_argument("rutaNC",help="(Obligatorio) Ruta donde estan los nc")
parser.add_argument("rutaRes", help = "Ruta donde se guardan las imagenes procesadas")
parser.add_argument("-t","--dt",help="(Opcional) Delta de t en segundos",default = 300,type=float)
parser.add_argument("-u","--umbral",help="(Opcional) Umbral de lluvia minima",default = 0.005,type=float)
parser.add_argument("-v","--verbose",help="Informa sobre la fecha que esta agregando", 
	action = 'store_true')
parser.add_argument("-s","--super_verbose",help="Imprime para cada posicion las imagenes que encontro",
	action = 'store_true')
parser.add_argument("-o","--old",help="Si el archivo a generar es viejo, y se busca es actualizarlo y no borrarlo",
	default = False)
parser.add_argument("-1","--hora_1",help="Hora inicial de lectura de los archivos",default= None )
parser.add_argument("-2","--hora_2",help="Hora final de lectura de los archivos",default= None )
parser.add_argument("-c","--save_class",help="Guarda los binarios del clasificado de lluvia",
	action = 'store_true')
parser.add_argument("-j","--save_escenarios",help="Guarda los binarios con los umbrales alto y bajo de la lluvia",
	action = 'store_true')

#lee todos los argumentos
args=parser.parse_args()
#-------------------------------------------------------------------------------------------------------------------------------------
#OBTIENE FECHAS Y DEJA ESE TEMA LISTO 
#-------------------------------------------------------------------------------------------------------------------------------------
#Obtiene las fechas por dias
datesDias = pd.date_range(args.fechaI, args.fechaF,freq='D')
a = pd.Series(np.zeros(len(datesDias)),index=datesDias)
a = a.resample('A').sum()
Anos = [i.strftime('%Y') for i in a.index.to_pydatetime()]

datesDias = [d.strftime('%Y%m%d') for d in datesDias.to_pydatetime()]

ListDays = []
ListRutas = []
for d in datesDias:
    try:
        L = glob.glob(args.rutaNC + d + '*.nc')
        ListRutas.extend(L)
        ListDays.extend([i[-23:-11] for i in L])
    except:
        print 'mierda'
#Organiza las listas de dias y de rutas
ListDays.sort()
ListRutas.sort()
datesDias = [dt.datetime.strptime(d[:12],'%Y%m%d%H%M') for d in ListDays]
datesDias = pd.to_datetime(datesDias)
#Obtiene las fechas por Dt
textdt = '%d' % args.dt
#Agrega hora a la fecha inicial
if args.hora_1 <> None:
        inicio = args.fechaI+' '+args.hora_1
else:
        inicio = args.fechaI
#agrega hora a la fecha final
if args.hora_2 <> None:
        final = args.fechaF+' '+args.hora_2
else:
        final = args.fechaF
datesDt = pd.date_range(inicio,final,freq = textdt+'s')
#Obtiene las posiciones de acuerdo al dt para cada fecha
PosDates = []
pos1 = [0]
for d1,d2 in zip(datesDt[:-1],datesDt[1:]):
        pos2 = np.where((datesDias<d2) & (datesDias>=d1))[0].tolist()
        if len(pos2) == 0:
                pos2 = pos1
        else:
                pos1 = pos2
        PosDates.append(pos2)


#-------------------------------------------------------------------------------------------------------------------------------------
#CARGADO DE LA CUENCA SOBRE LA CUAL SE REALIZA EL TRABAJO DE OBTENER CAMPOS
#-------------------------------------------------------------------------------------------------------------------------------------
#Carga la cuenca del AMVA
cuAMVA = wmf.SimuBasin(rute = args.cuenca)
cuConv = wmf.SimuBasin(rute = args.cuenca)
cuStra = wmf.SimuBasin(rute = args.cuenca)
cuHigh = wmf.SimuBasin(rute = args.cuenca)
cuLow = wmf.SimuBasin(rute = args.cuenca)

#si el binario el viejo, establece las variables para actualizar
if args.old:
    cuAMVA.rain_radar2basin_from_array(status='old',ruta_out= args.rutaRes)
    if args.save_class:
		cuConv.rain_radar2basin_from_array(status='old',ruta_out= args.rutaRes + '_conv')
		cuStra.rain_radar2basin_from_array(status='old',ruta_out= args.rutaRes + '_stra')
    if args.save_escenarios:
		cuHigh.rain_radar2basin_from_array(status='old',ruta_out= args.rutaRes + '_high')
		cuLow.rain_radar2basin_from_array(status='old',ruta_out= args.rutaRes + '_low')
#Itera sobre las fechas para actualizar el binario de campos
datesDt = datesDt.to_pydatetime()
for dates,pos in zip(datesDt[1:],PosDates):
	rvec = np.zeros(cuAMVA.ncells, dtype = float)
	if args.save_escenarios:
		rhigh = np.zeros(cuAMVA.ncells, dtype = float)
		rlow = np.zeros(cuAMVA.ncells, dtype = float)
	Conv = np.zeros(cuAMVA.ncells, dtype = int)
	Stra = np.zeros(cuAMVA.ncells, dtype = int)
	try:
		for c,p in enumerate(pos):
			#Lee la imagen de radar para esa fecha
			g = netCDF4.Dataset(ListRutas[p])
			RadProp = [g.ncols, g.nrows, g.xll, g.yll, g.dx, g.dx]                        
			#Agrega la lluvia en el intervalo 
			rvec += cuAMVA.Transform_Map2Basin(g.variables['Rain'][:].T/ (12*1000.0), RadProp) 
			if args.save_escenarios:
				rhigh += cuAMVA.Transform_Map2Basin(g.variables['Rhigh'][:].T / (12*1000.0), RadProp) 
				rlow += cuAMVA.Transform_Map2Basin(g.variables['Rlow'][:].T / (12*1000.0), RadProp) 
			#Agrega la clasificacion para la ultima imagen del intervalo
			ConvStra = cuAMVA.Transform_Map2Basin(g.variables['Conv_Strat'][:].T, RadProp)
			Conv = np.copy(ConvStra)
			Conv[Conv == 1] = 0; Conv[Conv == 2] = 1
			Stra = np.copy(ConvStra)
			Stra[Stra == 2] = 0 
			rvec[(Conv == 0) & (Stra == 0)] = 0
			if args.save_escenarios:
				rhigh[(Conv == 0) & (Stra == 0)] = 0
				rlow[(Conv == 0) & (Stra == 0)] = 0
			Conv[rvec == 0] = 0
			Stra[rvec == 0] = 0
			#Cierra el netCDFs
			g.close()
	except Exception, e:
		rvec = np.zeros(cuAMVA.ncells)
		if args.save_escenarios:
			rhigh = np.zeros(cuAMVA.ncells)
			rlow = np.zeros(cuAMVA.ncells)
		Conv = np.zeros(cuAMVA.ncells)
		Stra = np.zeros(cuAMVA.ncells)
	
	
	#rvec[ConvStra==0] = 0
	#rhigh[ConvStra==0] = 0
	#rlow[ConvStra==0] = 0
    #Escribe el binario de lluvia
	dentro = cuAMVA.rain_radar2basin_from_array(vec = rvec,
		ruta_out = args.rutaRes,
		fecha = dates-dt.timedelta(hours = 5),
		dt = args.dt,
		umbral = args.umbral)
	if args.save_escenarios:
		dentro = cuHigh.rain_radar2basin_from_array(vec = rhigh,
			ruta_out = args.rutaRes+'_high',
			fecha = dates-dt.timedelta(hours = 5),
			dt = args.dt,
			umbral = args.umbral)
		dentro = cuLow.rain_radar2basin_from_array(vec = rlow,
			ruta_out = args.rutaRes+'_low',
			fecha = dates-dt.timedelta(hours = 5),
			dt = args.dt,
			umbral = args.umbral)
	if dentro == 0: 
		hagalo = True
	else:
		hagalo = False
	#mira si guarda o no los clasificados
	if args.save_class:
		#Escribe el binario convectivo
		aa = cuConv.rain_radar2basin_from_array(vec = Conv,
			ruta_out = args.rutaRes+'_conv',
			fecha = dates-dt.timedelta(hours = 5),
			dt = args.dt,
			doit = hagalo)
	    #Escribe el binario estratiforme
		aa = cuStra.rain_radar2basin_from_array(vec = Stra,
			ruta_out = args.rutaRes+'_stra',
			fecha = dates-dt.timedelta(hours = 5),
			dt = args.dt,
			doit = hagalo)	
    #Opcion Vervose
	if args.verbose:
		print dates.strftime('%Y%m%d-%H:%M'), pos

#Cierrra el binario y escribe encabezado
cuAMVA.rain_radar2basin_from_array(status = 'close',ruta_out = args.rutaRes)
if args.save_class:
	cuConv.rain_radar2basin_from_array(status = 'close',ruta_out = args.rutaRes+'_conv')
	cuStra.rain_radar2basin_from_array(status = 'close',ruta_out = args.rutaRes+'_stra')
if args.save_escenarios:
	cuHigh.rain_radar2basin_from_array(status = 'close',ruta_out = args.rutaRes+'_high')
	cuLow.rain_radar2basin_from_array(status = 'close',ruta_out = args.rutaRes+'_low')
#Imprime en lo que va
if args.verbose:
        print 'Encabezados de binarios de cuenca cerrados y listos'
