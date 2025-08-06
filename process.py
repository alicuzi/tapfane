#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This script defines the main process where following tasks are called:
- training or testing is defined
- data is collected
- loaders are created
- training or testing is run with specified configuration of parameters defined
'''

__author__ = "Alice Cuzzucoli"
__copyright__ = "2025, Project ArcticPASSION, Institute of Atmospheric Pollution Research - National Research Council of Italy (CNR-IIA)"
__date__ = "2025/07/31"
__licence__ = ""
__status__ = "Production"

''' Python Libraries '''
import argparse
import os
import sys
import torch
import logging
import datetime
import traceback
import numpy as np

''' Local Libraries '''
from parameters import create_param_grid, Configuration
from data_manager.data_loader import collect_datasets, create_loaders
from training import Training
from test import Eval

start = {'year' : 2024,
         'month' : 1,
         'day' : 1}

end = {'year' : 2024,
       'month' : 12,
       'day' : 4}

def main():
    fix_seed = 2025
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    start_time = datetime.datetime.now()
    
    ''' Parser '''
    parser = argparse.ArgumentParser(description='Adapted Crossformer process')
    parser.add_argument('--task', type=str, default='train', help='definition of task (train,eval)')
    parser.add_argument('--start_date',type=dict,default=start)
    parser.add_argument('--end_date',type=dict,default=end)

    args = parser.parse_args()

    if args.task == 'train':
        param_grid = create_param_grid()
        
        for para in param_grid:
            config = Configuration(para)

            comb_train, comb_vali, comb_test = collect_datasets(config) 

            train_loader, vali_loader, test_loader = create_loaders(comb_train,
                                                                    comb_vali,
                                                                    comb_test,
                                                                    batch_size=config.batch_size)

            task = Training(config)
            task.train(train_loader, vali_loader)
            task.test(test_loader)

    elif args.task == 'eval':
        task = Eval(config)
        task.test(test_loader)
    else:
        print('No valid task')

    end_time = datetime.datetime.now()
    print(f"start time: {start_time}")
    print(f"end time: {end_time}")
    print(f"tot time: {end_time-start_time}")
    
    torch.cuda.empty_cache()

    return


if __name__ == "__main__":
    main()
