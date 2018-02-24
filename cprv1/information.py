#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
ESTACIONES = {
	u'user'						: [u'Usuario',None],
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
	u'l_tot_cauces'				: [u'Longitud total de los cauces',None]
	u'timestamp'				: [u'Fecha de creación de la base de datos',None],
	u'updated'					: [u'Fecha de la última modificación',None],
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
	'table' 					: None,
	'dbname'					:"cpr",
	'port'  					: 3306
                }
# argumentos por defecto para class SqlDb servidor de siata
REMOTE = {
	'host'  					:"localhost",
	'user'  					:"sample_user",
	'passwd'					:'s@mple_p@ss',
	'table' 					: None,
	'dbname'					:"cpr",
	'port'  					: 3306
			}
REDES = [
        'nivel', 'mocoa-nivel', 'Pluviografica', 'velocidad_superficial_rio',
        'mocoa-pluvio', 'mocoa-methiess','sirena', 'meteorologica',
        'humedad','meteorologica_thiess', 'pluviografica'
        ]
