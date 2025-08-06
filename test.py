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
import os
import time
import numpy as np
import matplotlib.pyplot as plt

''' Local Libraries '''
from model.crossformer import Crossformer
from utils.metrics import metric

class Eval(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        ''' build model '''
        self.pretrained_model = self._build_model().to(self.device)

        ''' upload weights '''
        self.pretrained_model.load_state_dict(torch.load(self.args.path_to_pretrained))
        # self.model = self.pretrained_model # temp variable
        # self.updated_model = self.pretrained_model # model to update with new training

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

    # def _load_model(self, model_path):
    #     '''
    #     load weights from pretrained model
    #     '''
    #     # self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, self.args.trained_model)))
    #     self.model.load_state_dict(torch.load(model_path))
    #     return 

    def test(self, test_loader):
        test_steps = len(test_loader)
        print(f"testing on {test_steps} samples")
        # print(self.model)

        self.model.eval()
        print("model loaded")

        preds = []
        trues = []
        metrics_all = []
        # metrics_cams = []
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
                # cams = batch_y[:,24:,:18].detach().cpu().numpy().transpose((0,2,1)).reshape(batch_y.shape[0],9,48) 

                # outputs_rescaled = (outputs * std[0,-1]) + mean[0,-1]
                # batch_y_rescaled = (batch_y.detach().cpu() * std[0,:]) + mean[0,:]
                # true_rescaled = batch_y_rescaled[:,:,-1]
                # # cams_rescaled = batch_y_rescaled[:,:,:18]

                batch_size = pred.shape[0]
                instance_num += batch_size
                # evaluate predicted values
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                # # evaluate cams
                # batch_cams = []
                # for i in range(cams.shape[1]):
                #     #print(i, cams[:,i,:])
                #     #print(metric(cams[:,i,:], true.detach().cpu().numpy()))
                #     batch_cams.append(np.array(metric(cams[:,i,:], true.detach().cpu().numpy()))* batch_size)
                # metrics_cams.append(batch_cams)
                
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # metrics_cams = np.stack(metrics_cams, axis = 0)
        # metrics_cams_mean = metrics_cams.sum(axis = 0) / instance_num

        # result save
        folder_path = '/home/alice/Desktop/ArcticPassion/Model_Training/test_results/' + self.args.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))
        # c_mae, c_mse, c_rmse, c_mape, c_mspe = metrics_cams_mean.transpose()
        # for i in range(cams.shape[1]):
        #     print('cams {} mse:{}, mae:{}'.format(i,c_mse[i], c_mae[i]))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'cams_metrics.npy', np.array([c_mae, c_mse, c_rmse, c_mape, c_mspe]))

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        print("test results saved")

        # # plot solution sample
        # plt.plot(true_rescaled[-1], label='true')
        # plt.plot(outputs_rescaled[-1].detach().cpu(), label='pred')
        # plt.plot(cams_rescaled[-1,:,0], label='chimere')
        # plt.plot(cams_rescaled[-1,:,1], label='dehm')
        # plt.legend()
        # plt.savefig(folder_path + self.args.setting +'.png', bbox_inches='tight')

        # self.log_results(cams, mse, mae, c_mse, c_mae)

        f = open(self.args.path_to_results_log + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.close()
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
            print('cams {} mse:{}, mae:{}'.format(i,c_mse[i], c_mae[i]))

        # save results
        folder_path = '/home/alice/Desktop/ArcticPassion/Model_Training/test_results/' + self.args.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+'cams_metrics.npy', np.array([c_mae, c_mse, c_rmse, c_mape, c_mspe]))
        print("cams metrics saved")

        f = open(self.args.path_to_results_log + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        for i in range(cams.shape[1]):
            f.write('cams {} mse:{}, mae:{}'.format(i,c_mse[i], c_mae[i]))
            f.write('\n')
        f.write('\n')
        f.close()
        print("cams metrics logged")


    def log_results(self, cams, mse, mae, c_mse, c_mae):    
        f = open(self.args.path_to_results_log + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        for i in range(cams.shape[1]):
            f.write('cams {} mse:{}, mae:{}'.format(i,c_mse[i], c_mae[i]))
            f.write('\n')
        f.write('\n')
        f.close()

        return