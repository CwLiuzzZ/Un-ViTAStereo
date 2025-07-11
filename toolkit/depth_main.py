from tqdm import tqdm
import numpy as np
import random
import torch
# import cv2
# from PIL import Image
# import argparse
# import json 
# import math
# import os
torch.set_float32_matmul_precision('high') #highest,high,medium
torch.backends.cudnn.benchmark = True # # Accelate training
# torch.autograd.set_detect_anomaly(True) # todo delete
from torch.utils.data import DataLoader

# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# CUDA_LAUNCH_BLOCKING=1


import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*exists and is not empty.")
warnings.filterwarnings("ignore", ".*logging on epoch level in distributed setting*")
warnings.filterwarnings("ignore", ".*RoPE2D, using a slow pytorch version instead")

import sys
sys.path.append('..')
from toolkit.data_loader.dataset_function import generate_file_lists
from toolkit.data_loader.dataloader import prepare_dataset,dataloader_customization,optimizer_customization
from toolkit.args.args_default import get_opts
from toolkit.torch_lightning.pl_modules_depth.depth_train import depth_trainer_func
from toolkit.torch_lightning.pl_modules_depth.depth_evaluate import depth_evaluate_func

# For reproducibility
torch.manual_seed(192)
torch.cuda.manual_seed(192)
np.random.seed(192)
random.seed(192)

def inference(hparams):

    ###################### 
    # prepare dataloader # 
    ###################### 
    hparams,aug_config,valid_aug_config = dataloader_customization(hparams)   
    file_path_dic = generate_file_lists(dataset = hparams.dataset,if_train=hparams.dataset_type=='train',method='gt',save_method=hparams.save_name)
    dataset,n_img = prepare_dataset(file_path_dic,aug_config=aug_config,inference_type=hparams.inference_type,save_method=hparams.save_name)
    hparams = optimizer_customization(hparams,n_img)
    if hparams.if_use_valid:
        valid_file_path_dic = generate_file_lists(dataset = hparams.val_dataset,if_train=hparams.val_dataset_type=='train',method='gt',save_method=hparams.save_name) 
        valid_dataset,_ = prepare_dataset(valid_file_path_dic,aug_config=valid_aug_config,inference_type = hparams.inference_type,save_method=hparams.save_name)
    else:
        valid_dataset = None

    ############################################
    # load model and select inference function #
    ############################################
    if hparams.inference_type == 'depth_train':
        inference = depth_trainer_func
    elif hparams.inference_type == 'depth_evaluate':
        inference = depth_evaluate_func
    ##########################
    # run inference function #
    ##########################
    if 'train' in hparams.inference_type:
        inference(hparams,dataset,valid_dataset)
    elif 'evaluate' in hparams.inference_type:
        test_dataloader = DataLoader(dataset, batch_size= 1, shuffle= False, num_workers= 1, drop_last=False)
        inference(hparams,test_dataloader)

# generate depth priors 
def depth_generate():
    hparams = get_opts() 
    hparams.inference_type = 'depth_evaluate'
    hparams.network = 'Depth' 
    hparams.dataset = 'KITTI2012+KITTI2015'
    return hparams

def train():
    hparams = get_opts() 
    hparams.devices = [0]
    hparams.epoch_size = 1
    hparams.num_workers = 2
    hparams.batch_size = 1
    hparams.inference_type = 'depth_train'
    hparams.network = 'ViTASIGEV' 
    hparams.pre_trained = False
    hparams.save_name = 'pre_train'
    hparams.dataset = 'KITTI2015'
    hparams.if_use_valid = True
    hparams.val_dataset = 'KITTI2012' 
    return hparams
    
# evaluate the Un-ViTAStereo on the KITTI benchmark
def evaluate():
    hparams = get_opts() 
    hparams.inference_type = 'depth_evaluate'
    hparams.network = 'ViTASIGEV' 
    hparams.dataset = 'KITTI2015_test+KITTI2012_test'
    hparams.ckpt_path = 'models/UnViTAStereo/unsupervised_benchmark.ckpt'
    hparams.save_name = 'Un-ViTAStereo'
    return hparams

if __name__ == '__main__':   
    
    hparams = evaluate()
    inference(hparams)
    
    # hparams = depth_generate()
    # inference(hparams)
    
    # hparams = train()
    # inference(hparams)
    