#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script contains functions to create datasets and loaders as inputs for the model:
- create nan df: creates a list of gaps to use for segmentation
- segment dataset: applies segmentation deleting large gaps (>=6 days)
- find bound dates: fine indices within dataset for train/vali/test split
- collect datasets: 
'''

__author__ = "Alice Cuzzucoli"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = "MIT"
__status__ = "Production"

''' Python Libraries '''
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd

''' Local Libraries '''
from data_manager.data_framer import StationDataset, StationDataset_shift
from utils.spatial import spatial_stats


# def collect_stations(meta, st_number):
#     # meta = pd.read_csv(path_to_meta)
#     # top_stations_meta = meta.sort_values(by=['NaN_%']).head(st_number)
#     # print(top_stations_meta['StationCode'])
#     ''' top stations '''
#     if st_number.isdigit():
#     	top_stations_meta = meta.loc[(meta['nans%']<=0.1) & (meta['longest_nan_gap']<=72)].head(st_number)
#         return list(top_stations_meta['code'])
    
#     else:
# 	    return list(meta['code'])

def create_nan_df(df):
     '''
     Creates dataframe where each entry is a gap and features are: 
     starting and ending time of gap, total length of gap
     '''

     ''' Create mask '''
     m = df['concentration'].shift(fill_value=False).notna()   # shift entries (populating with False empty values) and associate whether entry is NaN or not
     mask = df['concentration'].isna() | ~m   # either the elmt is NaN or the next is a NaN

     ''' Create aggregated dataframe '''
     nan_df = (df.groupby([m.cumsum()[mask]])
          .agg(Starting_Timestamp = ('time','min'),
               Ending_Timestamp = ('time','max'))
          .assign(Duration = lambda x: x['Ending_Timestamp'].sub(x['Starting_Timestamp']))
          .reset_index()
          .drop('concentration',axis=1)
        )
     
     return nan_df

def segment_dataset(config, name):
    ''' 
    read full dataset (interpolated) and gaps list
    data = [CAMS_d, CAMS_d+1, meteo_d, meteo_d+1, conc]
    gaps = [Starting_Timestamp, Ending_Timestamp, Duration]
    '''
    # read data
    data = pd.read_csv(config.path_to_data_folder + name + '_data.csv')
    gaps = create_nan_df(data) #pd.read_csv(config.path_to_gaps + name + '_gaps.csv')
    train_bound, test_bound = find_bound_dates(config) # find bounds for timeframes
    min_segment_len = 144 if config.shift == 0 else 168 # minimal length to construct Dataset object (__len__>0)

    # create large gaps sub-dataset
    large_gap_length = 6 # days
    lg_indices = []
    for row in range(len(gaps)):
        if int(gaps['Duration'].iloc[row].split()[0]) >= large_gap_length:
            lg_indices.append(row)
    large_gaps = gaps.iloc[lg_indices].reset_index(drop=True)

    segmented_data = [] # array to collect processed segments
    temp_segments = [] # temporary array for segments without large gaps
    # for each gap select data within end date of previous gap and start date of current gap
    for i in range(len(large_gaps)+1):
        if i == 0:
            temp_df = data.loc[(data['time'] < large_gaps['Starting_Timestamp'][i])]
        elif i == len(large_gaps):
            temp_df = data.loc[(data['time'] >= large_gaps['Ending_Timestamp'][i-1])]           
        else:
            temp_df = data.loc[(data['time'] >= large_gaps['Ending_Timestamp'][i-1]) & (data['time'] < large_gaps['Starting_Timestamp'][i])]
        if len(temp_df) >= min_segment_len: # extracted dataset must contain a vector including model input and output
            temp_segments.append(temp_df.reset_index(drop=True))

    # for each segments without large gaps check split timeframes
    for data_segment in temp_segments:
        subsegments = []
        first_date = pd.to_datetime(data_segment['time'].iloc[0])
        last_date = pd.to_datetime(data_segment['time'].iloc[-1]) 
        # 4 cases of intersections with split bounds: no intersection, intersection with train or test, intersection with both
        if (last_date <= train_bound) or (first_date >= test_bound) or ((first_date >= train_bound) and (last_date <= test_bound)): # data_segment is within one timeframe
            subsegments.append(data_segment)
        elif (first_date >= train_bound): # last_date is after train_bound AND first_date is before test_bound but they are not both in vali timeframe
            ts_idx = data_segment.loc[data_segment['time']==str(test_bound)].index[0] # dataset index of test bound date
            subsegments.append(data_segment.iloc[:ts_idx])
            subsegments.append(data_segment.iloc[ts_idx:])
        elif (last_date <= test_bound): # first date is before test bound but last date is not 
            tr_idx = data_segment.loc[data_segment['time']==str(train_bound)].index[0] # dataset index of train bound date
            subsegments.append(data_segment.iloc[:tr_idx])
            subsegments.append(data_segment.iloc[tr_idx:])
        else:
            tr_idx = data_segment.loc[data_segment['time']==str(train_bound)].index[0] # dataset index of train bound date
            ts_idx = data_segment.loc[data_segment['time']==str(test_bound)].index[0] # dataset index of test bound date
            subsegments.append(data_segment.iloc[:tr_idx])
            subsegments.append(data_segment.iloc[tr_idx:ts_idx])
            subsegments.append(data_segment.iloc[ts_idx:])

        # add to segment collection if at least a pair of input-output sequences can be extracted
        for sub in subsegments:
            if ((len(sub) >= min_segment_len) and (len(sub.loc[sub['concentration'].isna()==True])==0)): # extracted dataset must contain a vector including model input and output
                segmented_data.append(sub.reset_index(drop=True))
    return segmented_data

def find_bound_dates(config, station): #I_5_15986'):
    df = pd.read_csv(config.path_to_data_folder + station + '_data.csv')
    train_size = int(len(df)*config.train_test_split[0])
    vali_size = int(len(df)*config.train_test_split[1])
    test_size = int(len(df)*config.train_test_split[2])
    train_bound = df['time'].iloc[train_size] # dates up until train_bound used for training
    test_bound = df['time'].iloc[-test_size] # dates between train_bound and test_bound are for vali, test otherwise
    return pd.to_datetime(train_bound), pd.to_datetime(test_bound)

def collect_datasets(config):
    # read metadata file
    meta = pd.read_csv(config.path_to_meta)
    # collect stations codes
    stations = list(meta['code']) #collect_stations(meta, config.st_number)
    print(stations)

    # find spatial stats
    sp_stats = spatial_stats(meta)

    # define data setting
    if config.shift == 0:
        Dataset = StationDataset
    elif config.shift == 48:
        Dataset = StationDataset_shift

    # initialise datasets lists
    train_datasets = []
    vali_datasets = []
    test_datasets = []

    # create torch dataset for each station and collect in train/vali/test categories
    for name in stations:
        train_bound, test_bound = find_bound_dates(config) # find bounds for timeframes
        try:
            if (config.segmentation == True) and (meta['segmentation'].loc[meta['code']==name].values[0]==True):
                '''
                If dataset contains large gaps (including several missing values at the beginning, missed by interpolation)
                then the dataset is segmented in pieces which are then associated to train/vali/test
                given data bounds for split (calculated from the station with fewer nans).
                '''
                segments = segment_dataset(config, name) # segment dataset and collect segments in array
                print(f"Dataset {name} segmented in {len(segments)} parts")
                for data_segment in segments: # for each segment assess the timeframe, create torch Dataset and add to dataset collection
                    if (pd.to_datetime(data_segment['time'].iloc[-1]) < train_bound): # last date is within train timeframe
                        # train
                        train_dataset = Dataset(station_name=name,
                                                spatial_stats=sp_stats,
                                                config = config,
                                                flag='train',
                                                segment=data_segment)
                        train_datasets.append(train_dataset)
                    elif pd.to_datetime(data_segment['time'].iloc[-1]) < test_bound: #last date is not within train but within vali timeframe
                        # vali
                        vali_dataset = Dataset(station_name=name,
                                                spatial_stats=sp_stats,
                                                config = config,
                                                flag='vali',
                                                segment=data_segment)
                        vali_datasets.append(vali_dataset)
                    else: # last date is within test timeframe                        
                        # test
                        test_dataset = Dataset(station_name=name,
                                                spatial_stats=sp_stats,
                                                config = config,
                                                flag='test',
                                                segment=data_segment)
                        test_datasets.append(test_dataset)
                        print(f'segmented {name} len={len(test_dataset)}')

            else:  
                print(f"Dataset {name} considered in full length")  
                # train
                train_dataset = Dataset(station_name=name,
                                        spatial_stats=sp_stats,
                                        config = config,
                                        flag='train')
                train_datasets.append(train_dataset)

                # vali
                vali_dataset = Dataset(station_name=name,
                                        spatial_stats=sp_stats,
                                        config = config,
                                        flag='vali')
                vali_datasets.append(vali_dataset)

                # test
                test_dataset = Dataset(station_name=name,
                                        spatial_stats=sp_stats,
                                        config = config,
                                        flag='test')
                test_datasets.append(test_dataset)
                print(f'full {name} len={len(test_dataset)}')

        except Exception as err:
            print(err)
            print(f'No station {name} found')
            pass

    # concatenate datasets
    combined_train = ConcatDataset(train_datasets)
    combined_vali = ConcatDataset(vali_datasets)
    combined_test = ConcatDataset(test_datasets)

    return combined_train, combined_vali, combined_test


def create_loaders(train_ds, vali_ds, test_ds, batch_size=32):
    train_loader = DataLoader(  train_ds,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True )
    vali_loader = DataLoader(   vali_ds,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True )
    test_loader = DataLoader(   test_ds,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=True )

    return train_loader, vali_loader, test_loader

