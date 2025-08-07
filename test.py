#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script performs evaluation of PM10 forecast with the pretrained model
'''

__author__ = "Alice Cuzzucoli"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = ""
__status__ = "Production"

''' Python Libraries '''
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error


''' Local Libraries '''
from model.crossformer import Crossformer
from utils.metrics import metric
from utils.spatial import spatial_stats
from data_manager.data_framer import EvalStationDataset

class Eval(object):
    def __init__(self, args):
        self.args = args
        self.station = args.station_code
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.device = self._acquire_device()
        self.meta = pd.read_csv(args.root_path + args.meta_path)
        self.batch_size = 1
        self.path_to_pretrained = args.root_path + args.model_path + args.trained_model

        ''' build model '''
        self.pretrained_model = self._build_model().to(self.device)

        ''' upload weights '''
        self._load_model()

    def _acquire_device(self):
        if self.args.use_gpu:
            self.device = torch.device(f'cuda:{self.args.gpu}')
            print(f'use gpu: cuda:{self.args.gpu}')
        else:
            self.device = torch.device('cpu')
            print('use cpu')

    def _build_model(self):
        model = Crossformer(self.args.data_dim,
                            self.args.in_len,
                            self.args.out_len,
                            self.args.seg_len,
                            self.args.win_size,
                            self.args.factor,
                            self.args.d_model,
                            self.args.d_ff,
                            self.args.n_heads,
                            self.args.e_layers,
                            self.args.dropout,
                            self.device,
                            self.args.proj_layer,
                            self.args.embed
                            ).float()
        return model
    
    def _load_model(self):
        self.pretrained_model.load_state_dict(torch.load(self.path_to_pretrained,map_location=self.device))
        # print('Model Loaded')
        return self.pretrained_model
    
    def _load_data(self,date):
        sp_stats = spatial_stats(self.meta)
        data = EvalStationDataset(self.args, sp_stats,date)
        data_loader = DataLoader(data,
                                batch_size=self.batch_size,
                                shuffle=False,
                                drop_last=True)
        return data_loader

    def evaluate(self, args):
        try:
            ''' Create timestamps '''
            min_date = datetime.date(int(self.start_date['year']),int(self.start_date['month']),int(self.start_date['day']))
            max_date = datetime.date(int(self.end_date['year']),int(self.end_date['month']),int(self.end_date['day']))
            dates = pd.date_range(start=min_date, end=max_date,freq='d')

            for date in dates:
                try:
                    data_loader = self._load_data(date)
                    print("data loaded")

                    self.model.eval()
                    print("model loaded")

                    preds = []
                    trues = []

                    with torch.no_grad():

                        batch_x, batch_y, batch_geo, mean, std = next(iter(data_loader)) # only takes first batch of dataloader

                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_geo = batch_geo.float().to(self.device)

                        
                        outputs = self.model(batch_x, batch_geo)
                        ''' NOTE: dimension mismatch for 1 sample only, 
                        no need to squeeze predict_y in crossformer forward call '''

                        if self.args.scale:
                            preds = outputs.squeeze().detach().cpu().numpy()
                            trues = batch_y[:,:,-1].squeeze().detach().cpu().numpy()
                        else:
                            outputs_rescaled = (outputs * std[0,-1]) + mean[0,-1]
                            batch_y_rescaled = (batch_y.detach().cpu() * std[0,:]) + mean[0,:]
                            true_rescaled = batch_y_rescaled[:,:,-1]

                            preds = outputs_rescaled.squeeze().detach().cpu().numpy()
                            trues = true_rescaled.squeeze().detach().cpu().numpy()


                        mae = mean_absolute_error(trues, preds)
                        mse = mean_squared_error(trues, preds)

                        print(f'Adapted Crossformer mse:{mse}, mae:{mae}')

                        self.log_metrics(mse, mae, date)

                        self.log_results(trues, preds, date)
                        print('model results saved')
 
                except Exception as error:
                    print(f'''AdaptedCrossformer evaluation for station {self.station} for date {date} could NOT be completed,
                                    Error in DATES cycle occured: {error}''')
                    print(traceback.format_exc())
                    mae, mse = np.nan, np.nan
                    self.log_metrics(mse,mae,date)
                    pass

        except Exception as error:
            print("AdaptedCrossformer evaluation could NOT be completed")
            print(f"error occured: {error}")
            print(traceback.format_exc())
        return 
    
    def log_results(self, trues, preds, date):
        filename = self.args.station_code + '-' + str(date.date())
        if self.args.scale:
            filename += '-scaled'
        filepath = self.args.root_path + self.args.results_path + self.args.station_code +'/'
        filepath += self.args.station_code + '-' + str(date.date()) + '/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.save(filepath + filename + '-trues.npy',trues)
        np.save(filepath + filename + '-preds.npy',preds)
        
        return

    def log_metrics(self, mse, mae, date):
        filename = self.args.station_code + '-' + str(date.date())
        if self.args.scale:
            filename += '-scaled'
        filename += '-metrics.txt'
        filepath = self.args.root_path + self.args.results_path + self.args.station_code +'/'
        filepath += self.args.station_code + '-' + str(date.date()) + '/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        f = open(filepath + filename, 'w')
        f.write(self.args.setting + "  \n")
        f.write(f"model,mse,mae")
        f.write('\n')
        f.write(f"AdaptedCrossformer,{mse},{mae}")
        f.write('\n')
        f.write('\n')
        f.close()

        return
    
    def test(self, test_loader):
        test_steps = len(test_loader)
        print(f"testing on {test_steps} samples")

        self.model.eval()
        print("model loaded")

        preds = []
        trues = []
        metrics_all = []
        metrics_cams = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_geo, mean, std) in enumerate(test_loader):
                
                # process batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_geo = batch_geo.float().to(self.device)

                outputs = self.model(batch_x, batch_geo)

                pred = outputs
                true = batch_y[:,:,-1]
                cams = batch_y[:,24:,:18].detach().cpu().numpy().transpose((0,2,1)).reshape(batch_y.shape[0],9,48) 

                # outputs_rescaled = (outputs * std[0,-1]) + mean[0,-1]
                # batch_y_rescaled = (batch_y.detach().cpu() * std[0,:]) + mean[0,:]
                # true_rescaled = batch_y_rescaled[:,:,-1]
                # # cams_rescaled = batch_y_rescaled[:,:,:18]

                batch_size = pred.shape[0]
                instance_num += batch_size

                # evaluate predicted values
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                # evaluate cams
                batch_cams = []
                for i in range(cams.shape[1]):
                    #print(i, cams[:,i,:])
                    #print(metric(cams[:,i,:], true.detach().cpu().numpy()))
                    batch_cams.append(np.array(metric(cams[:,i,:], true.detach().cpu().numpy()))* batch_size)
                metrics_cams.append(batch_cams)
                
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        metrics_cams = np.stack(metrics_cams, axis = 0)
        metrics_cams_mean = metrics_cams.sum(axis = 0) / instance_num

        # result save
        folder_path = './eval_results/' + self.args.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print(f'mse:{mse}, mae:{mae}')
        c_mae, c_mse, c_rmse, c_mape, c_mspe = metrics_cams_mean.transpose()
        for i in range(cams.shape[1]):
            print(f'cams {i} mse:{c_mse[i]}, mae:{c_mae[i]}')

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'cams_metrics.npy', np.array([c_mae, c_mse, c_rmse, c_mape, c_mspe]))

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        print("test results saved")

        self.log_results(cams, mse, mae, c_mse, c_mae)

        print("test metrics logged")

        return  
    
    def cams_error(self, test_loader):
        trues = []
        metrics_cams = []
        instance_num = 0

        for i, (batch_x, batch_y, batch_geo, mean, std) in enumerate(test_loader):
            
            # process batch
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_geo = batch_geo.float().to(self.device)

            true = batch_y[:,:,-1]
            cams = batch_y[:,24:,:18].detach().cpu().numpy().transpose((0,2,1)).reshape(batch_y.shape[0],9,48) 

            # batch_y_rescaled = (batch_y.detach().cpu() * std[0,:]) + mean[0,:]
            # true_rescaled = batch_y_rescaled[:,:,-1]
            # cams_rescaled = batch_y_rescaled[:,:,:18]
            
            batch_size = batch_y.shape[0]
            instance_num += batch_size
            # evaluate cams
            batch_cams = []
            for i in range(cams.shape[1]):
                #print(i, cams[:,i,:])
                #print(metric(cams[:,i,:], true.detach().cpu().numpy()))
                batch_cams.append(np.array(metric(cams[:,i,:], true.detach().cpu().numpy()))* batch_size)
                metrics_cams.append(batch_cams)
                
                trues.append(true.detach().cpu().numpy())

        metrics_cams = np.stack(metrics_cams, axis = 0)
        metrics_cams_mean = metrics_cams.sum(axis = 0) / instance_num

        c_mae, c_mse, c_rmse, c_mape, c_mspe = metrics_cams_mean.transpose()
        for i in range(cams.shape[1]):
            print(f'cams {i} mse:{c_mse[i]}, mae:{c_mae[i]}')

        # save results
        folder_path = './test_results/' + self.args.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+'cams_metrics.npy', np.array([c_mae, c_mse, c_rmse, c_mape, c_mspe]))
        print("cams metrics saved")

        f = open(self.args.path_to_results_log + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        for i in range(cams.shape[1]):
            f.write(f'cams {i} mse:{c_mse[i]}, mae:{c_mae[i]}')
            f.write('\n')
        f.write('\n')
        f.close()
        print("cams metrics logged")


    def log_results(self, cams, mse, mae, c_mse, c_mae):    
        f = open(self.args.path_to_results_log + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        f.write(f'mse:{mse}, mae:{mae}')
        f.write('\n')
        for i in range(cams.shape[1]):
            f.write(f'cams {i} mse:{c_mse[i]}, mae:{c_mae[i]}')
            f.write('\n')
        f.write('\n')
        f.close()

        return