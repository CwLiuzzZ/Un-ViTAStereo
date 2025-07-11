import torch
from pytorch_lightning import LightningModule
import numpy as np

from toolkit.function.base_function import InputPadder
from toolkit.torch_lightning.lightning_function import schedule_select
from toolkit.function.models import DepthAny2
import os

class Depth(LightningModule):
    def __init__(self, hparams):
        super(Depth, self).__init__()
        self.save_hyperparameters()
        self.DepthAny2 = DepthAny2('vitl')

        
    def test_step(self, batch, batch_idx): 
        
        imgL   = batch['left'] # [B,3,H,W]
        imgR   = batch['right'] # [B,3,H,W]      

        padder = InputPadder(imgL.shape,14, mode = 'replicate')
        [imgL_depth,imgR_depth],_,_ = padder.pad(imgL, imgR)
        depth_L = self.DepthAny2.forward(imgL_depth)
        depth_R = self.DepthAny2.forward(imgR_depth)
        depth_L = padder.unpad(depth_L) # [B,H,W]
        depth_R = padder.unpad(depth_R) # [B,H,W]
        
        depth_save_dir = batch['save_dir_disp'][0].replace(self.hparams['hparams'].save_name,'DepthAnything')
        make_dir = '/'.join(depth_save_dir.split('/')[:-1])
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)
            print('make dir: {}'.format(make_dir))
        L_depth_path = depth_save_dir.replace('.npy','_depthL.npy')
        R_depth_path = depth_save_dir.replace('.npy','_depthR.npy')
        
        depth_L_np = depth_L.squeeze().detach().cpu().numpy()
        depth_R_np = depth_R.squeeze().detach().cpu().numpy()
        np.save(L_depth_path,depth_L_np)
        np.save(R_depth_path,depth_R_np)
        return 
        
        
    