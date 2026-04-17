#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script contains the Training class with training, validation and test functions.
Within the class, model with specific configuration is loaded, loss and optimiser are chosen,
checkpoints are saved and results are logged.
'''

__author__ = "Alice Cuzzucoli, Ilaria Crotti, Srdjan Dobricic and Antonello Pasini"
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
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

class Training(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'use gpu: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('use cpu')
        return device
    
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

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim   

    def _select_criterion(self):
        # criterion =  nn.MSELoss()
        # criterion = nn.L1Loss()
        criterion = nn.HuberLoss()
        return criterion 

    def train(self,train_loader,vali_loader):
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train() # set in training mode
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_geo, mean, std) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
            
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_geo = batch_geo.float().to(self.device)


                outputs = self.model(batch_x, batch_geo)
                
                # optimisation
                pred = outputs
                #true = batch_y # for target value only
                true = batch_y[:,:,-1]

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                # end epoch cycle

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time}")
            train_loss = np.average(train_loss)

            # Validation
            self.model.eval() # set in evaluation mode
            vali_loss = []

            with torch.no_grad(): # no optimisation step

                for i, (batch_x,batch_y, batch_geo, mean, std) in enumerate(vali_loader):
  
                    # process batch
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_geo = batch_geo.float().to(self.device)

                    outputs = self.model(batch_x, batch_geo)

                    # original values    
                    # outputs_rescaled = combined_vali.inverse_transform(outputs)
                    # batch_y_rescaled = combined_vali.inverse_transform(batch_y)

                    # calculate loss
                    pred = outputs
                    true = batch_y[:,:,-1]
                    loss = criterion(pred.detach().cpu(), true.detach().cpu())
                    vali_loss.append(loss.detach().item())

            vali_loss = np.average(vali_loss)
            self.model.train()

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            # check early stopping and save checkpoint
            path_to_checkpoint = self.args.path_to_checkpoints + self.args.setting
            if not os.path.exists(path_to_checkpoint):
                os.makedirs(path_to_checkpoint)
            early_stopping(vali_loss, self.model, path_to_checkpoint)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)  
            # end epochs cycle

        # load best checkpoint
        best_model_path = self.args.path_to_checkpoints + self.args.setting + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.args.path_to_checkpoints + self.args.setting + '/' + 'checkpoint.pth')

        return self.model
    
    def test(self, test_loader):
        test_steps = len(test_loader)
        print(f"testing on {test_steps} samples")

        self.model.eval()
        print("model loaded")

        # initialise results
        preds = []
        trues = []
        metrics_cams = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad(): # no optimisation step
            for i, (batch_x, batch_y, batch_geo, mean, std) in enumerate(test_loader):
                
                # process batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_geo = batch_geo.float().to(self.device)

                outputs = self.model(batch_x, batch_geo)

                pred = outputs
                true = batch_y[:,:,-1]
                cams = batch_y[:,24:,:18].detach().cpu().numpy().transpose((0,2,1)).reshape(batch_y.shape[0],9,48) 


                batch_size = pred.shape[0]
                instance_num += batch_size
                # evaluate predicted values
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

                batch_cams = []
                for i in range(cams.shape[1]):
                    batch_cams.append(np.array(metric(cams[:,i,:], true.detach().cpu().numpy()))* batch_size)
                metrics_cams.append(batch_cams)

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        metrics_cams = np.stack(metrics_cams, axis = 0)
        metrics_cams_mean = metrics_cams.sum(axis = 0) / instance_num

        # result save
        folder_path = self.args.path + self.args.path_to_results + self.args.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

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
        # f = open(self.args.path_to_results + self.args.filename, 'a')
        # f.write(self.args.setting + "  \n")
        # f.write(f'mse:{mse}, mae:{mae}')
        # f.write('\n')
        # f.close()
        print("test metrics logged")

        return  

    def log_results(self, cams, mse, mae, c_mse, c_mae):    
        f = open(self.args.path_to_results + self.args.filename, 'a')
        f.write(self.args.setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        for i in range(cams.shape[1]):
            f.write('cams {} mse:{}, mae:{}'.format(i,c_mse[i], c_mae[i]))
            f.write('\n')
        f.write('\n')
        f.close()

        return

