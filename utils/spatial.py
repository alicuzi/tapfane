#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script defines how spatial features are collected and rescaled to fit the Geometric Dimension-Wise Embedding
'''

__author__ = "Alice Cuzzucoli, Ilaria Crotti, Srdjan Dobricic and Antonello Pasini"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = ""
__status__ = "Production"

''' Python Libraries '''
import numpy as np
import pandas as pd


def spatial_stats(meta):

    # latitude stats
    lat_mean = meta['Latitude'].values.mean()
    lat_range = meta['Latitude'].values.max() - meta['Latitude'].values.min()

    # longitude stats
    lon_mean = meta['Longitude'].values.mean()
    lon_range = meta['Longitude'].values.max() - meta['Longitude'].values.min()

    # altitude stats
    alt_mean = meta['Altitude'].values.mean()
    alt_range = meta['Altitude'].values.max() - meta['Altitude'].values.min()

    spatial_stats = {   'lat_m' : lat_mean,
                    'lat_r' : lat_range,
                    'lon_m' : lon_mean,
                    'lon_r' : lon_range,
                    'alt_m' : alt_mean,
                    'alt_r' : alt_range
                    }

    return spatial_stats


class FeatureScaler():
    def __init__(self, mean, range):
        self.mean = float(mean)
        self.range = float(range)
    
    def transform(self, data):
        return (data - self.mean) / self.range - 0.5
    
    def inverse_transform(self, data):
        return ((data + 0.5) * self.range) + self.mean


def space_features(geocoords, stats):
    LatScaler = FeatureScaler(mean = stats['lat_m'], range = stats['lat_r'])
    LonScaler = FeatureScaler(mean = stats['lon_m'], range = stats['lon_r'])
    AltScaler = FeatureScaler(mean = stats['alt_m'], range = stats['alt_r'])

    # return np.vstack([LatScaler.transform(geocoords['Latitude']),
    #                  LonScaler.transform(geocoords['Longitude']),
    #                  AltScaler.transform(geocoords['Altitude']),
    #                  ]).transpose(1,0)
    return np.vstack([LatScaler.transform(geocoords['latitude']),
                     LonScaler.transform(geocoords['longitude']),
                     AltScaler.transform(geocoords['altitude']),
                     ]).transpose(1,0)