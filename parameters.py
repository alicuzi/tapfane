#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script manages model hyperparamenters configuration and paths needed for the process.
Multiple choices of hyperparameters can be selected to perform hyperparameter tuning.
'''

__author__ = "Alice Cuzzucoli"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = "MIT"
__status__ = "Production"

import itertools
import datetime


''' Trials '''
embedding_type = ['GDSW']
seq_len_choices = [96]
pred_len_choices = [48]
label_len_choices = [24]
proj_layer = [True]
data_shift_choices = [48] # 0=normal data, 48=shifted data
factor_choices =[6]
win_size_choices = [2]
input_dimension_choices = [31] # 31 all features, 19 no meteo
e_layers_choices = [3]
d_layers_choices = [1]
n_heads_choices = [4]
d_model_choices = [256]
d_ff_choices = [512]
dropout_choices = [0.2]
learning_rate_choices = [1e-4]
lr_adjust_choices = ['type1']
patience_choices = [3]
epochs_choices = [20]
batch_size_choices = [32]

''' Hyperparameter Tuning Grid '''
# embedding_type = ['DSW','GDSW']
# seq_len_choices = [48, 72, 96, 120]
# pred_len_choices = [48]
# label_len_choices = [6, 12, 24, 48]
# proj_layer = [True, False]
# data_shift_choices = [0, 48]
# factor_choices =[5,6,7,8,9,10]
# win_size_choices = [2]
# input_dimension_choices = [31]
# e_layers_choices = [2, 3, 4]
# d_layers_choices = [1]
# n_heads_choices = [3, 4, 5]
# d_model_choices = [128, 256, 512]
# d_ff_choices = [256, 512, 1024]
# dropout_choices = [0, 0.1, 0.2]
# learning_rate_choices = [1e-2, 1e-4]
# lr_adjust_choices = ['type1']
# patience_choices = [3]
# epochs_choices = [20]
# batch_size_choices = [32]

features = list([embedding_type,
                seq_len_choices,
                pred_len_choices,
                label_len_choices,
                proj_layer,
                data_shift_choices,
                factor_choices,
                win_size_choices,
                input_dimension_choices,
                e_layers_choices,
                d_layers_choices,
                n_heads_choices,
                d_model_choices,
                d_ff_choices,
                dropout_choices,
                learning_rate_choices,
                lr_adjust_choices,
                patience_choices,
                epochs_choices,
                batch_size_choices
                ])

feature_names = list(['embed',
                     'seq_len',
                     'pred_len',
                     'label_len',
                     'proj_layer',
                     'data_shift',
                     'factor',
                     'win_size',
                     'data_dim',
                     'e_layers',
                     'd_layers',
                     'n_heads',
                     'd_model',
                     'd_ff',
                     'dropout',
                     'learning_rate',
                     'lradj',
                     'patience',
                     'epochs',
                     'batch_size'
                    ])

def create_param_grid():
    feat_combs = []
    for comb in list(itertools.product(*features)):
        d = {}
        for name, f in zip(feature_names, comb):
            d.update({name:f})
        feat_combs.append(d)

    return feat_combs

class Configuration(object):
    def __init__(self, features):
        self.embed = features['embed']
        self.in_len = features['seq_len']
        self.out_len = features['pred_len']
        self.seg_len = features['label_len']
        self.proj_layer = features['proj_layer']
        self.shift = features['data_shift']
        self.factor = features['factor']
        self.win_size = features['win_size']
        self.data_dim = features['data_dim']
        self.e_layers = features['e_layers']
        self.d_layers = features['d_layers']
        self.n_heads = features['n_heads']
        self.d_model = features['d_model']
        self.d_ff = features['d_ff']
        self.dropout = features['dropout']
        self.learning_rate = features['learning_rate']
        self.lradj = features['lradj']
        self.patience = features['patience']
        self.train_epochs = features['epochs']
        self.batch_size = features['batch_size']

        self.st_number = '-training'

        self.train_test_split = [0.5, 0.25, 0.25]
        # self.train_test_split = [0.7, 0.1, 0.2]

        self.segmentation = True

        self.path = '' # root path
        self.path_to_meta = './stations_metadata.csv'
        self.path_to_checkpoints = './checkpoints/'
        self.path_to_results = './results/'
        self.path_to_data_folder = './training_data/'
        self.path_to_pretrained = './pretrained_model/CF-GDSW_stall_shift48_proj-True_dim31_inlen96_outlen48_seglen24_win2_f6_dmodel256_dff512_nheads4_elayers3_drop0.2_ep20_p3_0.5-0.25-0.25split_segm-True_huber'

        # define setting
        setting = f"CF-{self.embed}_st{self.st_number}_shift{self.shift}_proj-{self.proj_layer}_dim{self.data_dim}"
        setting += f"_inlen{self.in_len}_outlen{self.out_len}_seglen{self.seg_len}"
        setting += f"_win{self.win_size}_f{self.factor}_dmodel{self.d_model}_dff{self.d_ff}_nheads{self.n_heads}"
        setting += f"_elayers{self.e_layers}_drop{self.dropout}_ep{self.train_epochs}_p{self.patience}"
        setting += f"_{self.train_test_split[0]}-{self.train_test_split[1]}-{self.train_test_split[2]}split"
        setting += f"_segm-{self.segmentation}"
        print(setting)

        self.setting = setting
        self.filename = f'{datetime.datetime.date()}.txt'

        self.use_gpu = True
        self.gpu = 0

