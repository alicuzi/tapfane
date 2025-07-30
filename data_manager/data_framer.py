#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script contains Dataset definition for each station.
Spatial features are collected from metadata is Geometric Dimension-Wise Embedding is involved.
'''

__author__ = "Alice Cuzzucoli"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = ""
__status__ = "Production"

''' Python Libraries '''
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch


''' Local Libraries '''
from utils.spatial import space_features


class StationDataset(torch.utils.data.Dataset):
    def __init__(self, station_name,
                 spatial_stats,
                 config,
                 flag = 'train') -> None:
        super().__init__()

        self.spatial_stats = spatial_stats
        self.name = station_name
        self.embed = config.embed
        self.path = config.path_to_data_folder
        self.path_to_meta = config.path_to_meta


        ''' Dataset Type '''
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]

        ''' Set sequence lenghts'''
        self.seq_len = config.in_len
        self.pred_len = config.out_len

        ''' Set Scaler '''
        self.scaler = StandardScaler()

        ''' Read Data '''
        self.__read_data__()

    def __read_data__(self):

        ''' Retrieve Dataset '''
        dataset = pd.read_csv(self.path + self.name + '_data.csv')

        ''' Define Data Split '''
        train_size = int(len(dataset) * 0.5)
        test_size = int(len(dataset) * 0.25)
        vali_size = len(dataset) - train_size - test_size

        ''' Define bounds for sequences '''
        low_bound = [0, train_size - self.seq_len, len(dataset) - test_size - self.seq_len]
        up_bound = [train_size, train_size + vali_size, len(dataset)]
        border1 = low_bound[self.set_type]
        border2 = up_bound[self.set_type]

        ''' Scale Data '''
        train_data = dataset.iloc[low_bound[0]:up_bound[0],1:]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(dataset.iloc[:,1:].values) # array

        ''' Store stats '''
        ''' Standard scaler '''
        self.mean = self.scaler.mean_
        self.std = self.scaler.scale_

        ''' Store data '''
        self.data_x = data[border1:border2]
        #self.data_y = data[border1:border2,0] # take target feature only
        self.data_y = data[border1:border2] # take all features and select target feature in the optimisation step

        ''' Space features '''	
        if (self.embed == 'GDSW'):
            metadata = pd.read_csv(self.path_to_meta)
            geo = metadata[['longitude','latitude','altitude']].loc[metadata['code'] == self.name]
            # define and store spacefeatures
            self.spacefeatures = space_features(geo,self.spatial_stats)
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if (self.embed == 'GDSW'):
            return seq_x, seq_y, self.spacefeatures, self.mean, self.std    
            
        elif (self.embed == 'DSW'):
            return seq_x, seq_y, None, None, None     
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class StationDataset_shift(torch.utils.data.Dataset):
    def __init__(self, station_name,
                 spatial_stats,
                 config,
                 flag = 'train',
                 segment = []) -> None:
        super().__init__()

        self.spatial_stats = spatial_stats
        self.name = station_name
        self.embed = self.embed
        self.path = config.path_to_data_folder
        self.path_to_meta = config.path_to_meta

        self.split = config.train_test_split

        self.segment = segment # portion of dataset in case of segmentation

        ''' Dataset Type '''
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]

        ''' Set sequence lenghts'''
        self.seq_len = config.in_len
        self.pred_len = config.out_len

        ''' Set Scaler '''
        self.scaler = StandardScaler()

        ''' Read Data '''
        self.__read_data__()

    def __read_data__(self):

        ''' Retrieve Dataset '''
        dataset = pd.read_csv(self.path + self.name + '_data.csv')

        ''' Define Data Split '''
        train_size = int(len(dataset) * self.split[0])
        test_size = int(len(dataset) * self.split[2])
        vali_size = len(dataset) - train_size - test_size

        ''' Define bounds for sequences '''
        low_bound = [0, train_size - self.seq_len, len(dataset) - test_size - self.seq_len]
        up_bound = [train_size, train_size + vali_size, len(dataset)]
        if len(self.segment): # if dataset is segmented (len!=0), take the whole segment
            border1 = 0
            border2 = len(self.segment)
        else: # if dataset is taken in full, subdivide by task
            border1 = low_bound[self.set_type]
            border2 = up_bound[self.set_type]

        ''' Scale Data '''
        train_data = dataset.iloc[low_bound[0]:up_bound[0],1:]
        self.scaler.fit(train_data.values)
        if len(self.segment):
            data = self.scaler.transform(self.segment.iloc[:,1:].values) # array
        else:        
            data = self.scaler.transform(dataset.iloc[:,1:].values) # array

        ''' Store stats '''
        ''' Standard scaler '''
        self.mean = self.scaler.mean_
        self.std = self.scaler.scale_

        ''' Shift features '''
        #shift data_x so that cams+meteo are shifted forward
        self.data_x = np.hstack([
            data[border1+24:border2,:-1], # all but last feature (concentration) are shifted forward
            np.expand_dims(data[border1:border2-24,-1],axis=1)]) # last feature (concentration) is cut to fit dataset dim
        #self.data_y = data[border1:border2,0] # take target feature only
        self.data_y = data[border1:border2] # take all features and select target feature in the optimisation step

        ''' Space features '''	
        if (self.embed == 'GDSW'):
            metadata = pd.read_csv(self.path_to_meta)
            geo = metadata[['longitude','latitude','altitude']].loc[metadata['code'] == self.name]
            # define and store spacefeatures
            self.spacefeatures = space_features(geo,self.spatial_stats)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if (self.embed == 'GDSW'):
            return seq_x, seq_y, self.spacefeatures, self.mean, self.std    
            
        elif (self.embed == 'DSW'):
            return seq_x, seq_y, None, None, None 
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def name(self):
        return self.name
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
