#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

def read_csv(path,datetime_index = False,*args,**kwargs):
    '''
    pandas.read_csv() customized
    Parameters
    ----------
    path           : csv path
    datetime_index  :
    *args,**kwargs : arguments to pass
    Returns
    ----------
    pandas DataFrame
    '''
    df = pd.read_csv(path,index_col=0,*args,**kwargs)
    if datetime_index:
        df.index = df.index.to_datetime()
    return df

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


def filter_index(index):
    '''
    Selects index by good datetime convertion
    Parameter
    ---------
    index : pandas index object
    Returns
    ---------
    (good_date, bad_date) lists
    '''
    good_date = []; bad_date = []
    for i in index:
        try:
            pd.to_datetime(i)
            good_date.append(i)
        except ValueError:
            bad_date.append(i)
            pass
    return good_date,bad_date

def curva_duracion(df,bins=50):
    '''
    Estimates duration flow curve
    '''
    a,b = np.histogram(df,bins=bins)
    x = 100-np.cumsum(a/np.sum(np.array(a,dtype=float))*100.)
    y = (b[1:] + b[:-1] )/2.
    x[0] = 99.9
    return x,y

def make_colormap(seq):
    '''
    Creates colormap from colormap sequence
    seq : colormap sequence (rgb colors and boundaries)
    Parameters
    ----------
    Example
    ----------
    c = mcolors.ColorConverter().to_rgb
    cm = make_colormap([c('#D9E5E8'),0.20,c('green'),0.4,c('orange'),0.60,c('red'),0.80,c('indigo')])
    '''
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

def round_time(time,each=5.0):
    '''
    Gets the date near a freq = 'each' minutes dates
    '''
    redondeo = int(math.ceil(time.minute / each)) * each
    return pd.to_datetime((time + datetime.timedelta(minutes = redondeo-time.minute)).strftime(self.str_date_format))

def get_area(x,y):
    '''Calcula las areas y los caudales de cada
    una de las verticales, con el metodo de mid-section
    Input:
    x = Distancia desde la banca izquierda, type = numpy array
    y = Produndidad
    Output:
    area = Area de la subseccion
    Q = Caudal de la subseccion
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
