#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
ESTACIONES = {
	u'codigo'					: [u'Código','Codigo'],
	u'nombre'					: [u'Nombre','NombreEstacion',],
	u'direccion'				: [u'Dirección','DireccionCalleCarrera',],
	u'tipo_sensor'				: [u'Tipo de sensor','N'],
	u'vel_sup'					: [u'Código estación de velocidad superficial',None],
    u'area'						: [u'Área de a cuenca',None],
	u'subcuenca'				: [u'Subcuenca','Subcuenca'],
	u'barrio'					: [u'Barrio','Barrio'],
	u'source'					: [u'Tabla remota',None],
	u'l_cuenca'					: [u'Longitud de la cuenca',None],
    u'longitud_basin'           : [u'Longitud trazado cuenca',None],
	u'fecha_instalacion'		: [u'Fecha de instalación','FechaInstalacion'],
	u'h_cauce_max'				: [u'Longitud máxima del cauce',None],
	u'pend_cuenca'				: [u'Pendiente promedio de la cuenca',None],
	u'hmax'						: [u'Altura máxima del cauce',None],
	u'kml_path'					: [u'Ruta del kml',None],
	u'l_cauce'					: [u'Longitud del cauce',None],
	u'perimetro'				: [u'Perímetro de la cuenca',None],
	u'municipio'				: [u'Municipio','Ciudad'],
	u'hmean'					: [u'Altitud promedio',None],
	u'flag_modelo_estadistico'	: [u'Si tiene modelo estadístico',None],
    u'dpx'                      : [u'Resolucion de la cuenca',None],
	u'camara_path'				: [u'Ruta de las imágenes en remoto',None],
	u'rain_path'				: [u'Ruta de binario con lluvia ',None],
	u'pic_path'					: [u'Ruta de las imágenes',None],
	u'latitud'					: [u'Latitud','Latitude'],
	u'polygon_path'				: [u'Ruta del polígono de la cuenca',None],
	u'centro_x'					: [u'Centro x',None],
	u'centro_y'					: [u'Centro y',None],
	u'pluvios'					: [u'Estaciones de lluvia dentro de la cuenca',None],
    u'red'                      : [u'Red','Red'],
	u'telefono_contacto'		: [u'Teléfono del contacto',None],
	u'offset_old'				: [u'Offset viejo', 'offsetN',],
	u'flag_modelo_wmf'			: [u'Si tiene modelo físico',None],
	u'net_path'					: [u'Ruta del shape, drenaje',None],
	u'x_sensor'					: [u'Distancia del sensor al eje en x',None],
    u'latitud_basin'            : [u'Latitud trazado cuenca',None],
	u'sirena'					: [u'Si tiene sirena',None],
	u'offset'					: [u'Offset dinámico',None],
	u'estado'					: [u'Estado','estado'],
	u'clase'					: [u'Clase que controla el objeto en cpr',None],
	u'longitud'					: [u'Longitud','Longitude'],
    u'slug'						: [u'Slug',None],
	u'nombre_contacto'			: [u'Nombre del contacto',None],
	u'hmin'						: [u'Altitud mínima de la cuenca',None],
	u'pend_cauce'				: [u'Pendiente promedio del cauce',None],
	u'nc_path'					: [u'Ruta del netcdf de la cuenca .nc',None],
	u'stream_path'				: [u'Ruta del shape, cauce aguas abajo',None],
	u'n1'						: [u'Nivel de riesgo 1','action_level'],
	u'n2'						: [u'Nivel de riesgo 2','minor_flooding'],
	u'n3'						: [u'Nivel de riesgo 3','moderate_flooding'],
	u'n4'						: [u'Nivel de riesgo 4','major_flooding'],
	u'l_tot_cauces'				: [u'Longitud total de los cauces',None],
	u'timestamp'				: [u'Fecha de creación de la base de datos',None],
	u'updated'					: [u'Fecha de la última modificación',None],
	u'usr'					: [u'Usuario que modifica',None]
	}

HIDRAULICA = {
	'id_aforo'					: u'Identificador principal',
	'id_estacion_asociada'		: u'Código del sensor de nivel asociado',
	'dispositivo'				: u'Dispositivo de medida',
	'fecha'						: u'Fecha del procedimiento',
	'ancho_superficial'			: u'Ancho superficial',
	'caudal_medio'				: u'Caudal total',
	'error_caudal'				: u'Error caudal',
	'velocidad_media'			: u'Velocidad promedio',
	'velocidad_superficial'		: u'Velocidad superficial',
	'perimetro'					: u'Perímetro mojado',
	'area_total'				: u'Área mojada',
	'altura_media'				: u'Profundidad media',
	'radio_hidraulico'			: u'Radio hidráulico',
	'flag_izquierda'			: u'Inicio mirando aguas abajo',
	'flag_ubicacion'			: u'Si es cerca del sensor de nivel',
	'source'					: u'Si es siata, redrio u otro',
	'observacion'				: u'Descripción del sitio',
	'calidad'					: u'Calidad del dato',
	'xsensor'					: u'Distancia del sensor al eje referencia en x',
	'offset'					: u'Offset',
	'updated'					: u'Fecha de la última modificación',
	'codigo'					: u'Código de la estación',
	'flag_profundidades'		: u'Si se midieron profundidades en campo',
	'tipo_aforo'				: u'Aforo por vadeo, suspención, o por curva',
	'flag_flujo_dividido'	    : u'Si el aforo se hace en flujo dividido',
	'n1'						: u'Nivel de riesgo 1',
	'n2'						: u'Nivel de riesgo 2',
	'n3'						: u'Nivel de riesgo 3',
	'n4'						: u'Nivel de riesgo 4',
	'pic_path'					: u'Ruta de las imágenes',
	'user'						: u'Usuario que modifica',
			}
# argumentos por defecto para class SqlDb servidor local
LOCAL = {
	'host'  					:"localhost",
	'dbname'					:"cpr",
	'port'  					: 3306
                }
# argumentos por defecto para class SqlDb servidor de siata
REMOTE = {
	'host'  					: "192.168.1.74",
	'user'  					:"siata_Consulta",
	'passwd'					:"si@t@64512_C0nsult4",
	'table' 					: 'estaciones',
	'dbname'					:"siata",
	'port'  					: 3306
			}


REDES = [
        'nivel', 'mocoa-nivel', 'Pluviografica', 'velocidad_superficial_rio',
        'mocoa-pluvio', 'mocoa-methiess','sirena', 'meteorologica',
        'humedad','meteorologica_thiess', 'pluviografica'
        ]
GEOPARAMETERS = {
				'Area[km2]'               : u'area',
				'Centro_[X]'              : u'centro_x',
				'Centro_[Y]'              : u'centro_y',
				'H Cauce_Max [m]'         : u'h_cauce_max',
				'Hmax_[m]'                : u'hmax',
				'Hmean_[m]'               : u'hmean',
				'Hmin_[m]'                : u'hmin',
				'Long_Cau [km]'           : u'l_cauce',
				'Long_Cuenca [km]'        : u'l_cuenca',
				'Long_tot_cauces[km]'     : u'l_tot_cauces',
				'Pend_Cauce [%]'          : u'pend_cauce',
				'Pend_Cuenca [%]'         : u'pend_cuenca',
				'Perimetro[km]'           : u'perimetro'
				}

codigos = [ 128,108,245,109,106,186,124,135,247,140,96,101,246,92,94,245,238,251,1014,
            1013,260,158,182,93,239,90,104,143,183,240,99,91,115,116,134,152,166,179,
            155,236,173,178,196,195,259,268,98,272,273 ]

x_sensor = [ 8.0,4.23,11.66,3.5,17.0,5.75,12.6,3.08,3.0,24.0,4.2,0.8,4.2,1.3,12.11,11.66,
            12.54,4.1,6.88,8.74,21.0,3.87,2.45,31.17,6.4,11.6,2.55,4.66,2.8,1.5,21.0,18.95,
            1.55,8.21,2.5,3.0,2.0,8.0,5.5,14.0,2.0,1.5,29.8,3.92,2.42,4.4,5.0,5.74,3.18 ]

X_SENSOR = dict(zip(codigos,x_sensor))
