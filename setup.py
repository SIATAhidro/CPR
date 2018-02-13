#!/usr/bin/env python
import os
from numpy.distutils.core import setup, Extension

setup(
    name='hidrocprV1',
    version='0.0.1',
    author='Hidro SIATA',
    author_email='hidrosiata@gmail.com',
    packages=['cprv1'],
    package_data={'cpr':['cprv1.py','static.py','bookplots.py']},
    url='https://github.com/SIATAhidro/CPR.git',
    license='LICENSE.txt',
    description='Consultas-Plots y Reportes',
    long_description=open('README.md').read(),
    install_requires=[ ],
    )
