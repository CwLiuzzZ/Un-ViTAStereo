import torch.nn as nn
import torch
from YoYoModel.ViTAS.args.ViTAS_args import config_ViTASIGEV_args
from YoYoModel.ViTAS.ViTAS_model import ViTASBaseModel
from YoYoModel.ViTASIGEV.igev.igev_model import IGEVStereo,config_IGEV_args
from collections import OrderedDict
# import time

def get_parameter_number(model,name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, ' Total:', total_num, 'Trainable:', trainable_num)
    # return {'Total': total_num, 'Trainable': trainable_num}
    
class ViTASIGEVModel(nn.Module):
    def __init__(self,ViTAS_dic):
        super().__init__()
        self.ViTAS = self.load_ViTAS(ViTAS_dic)
        get_parameter_number(self.ViTAS,'ViTAS')
        self.igev = self.load_IGEV()

    def load_ViTAS(self,ViTAS_dic):
        args = config_ViTASIGEV_args(ViTAS_dic)
        model = ViTASBaseModel(**vars(args))
        return model
    
    def load_IGEV(self):
        igev_args = config_IGEV_args()
        model = IGEVStereo(igev_args)                
        return model
        
    def forward(self, img1, img2, iters, test_mode=False):  
        # if not '16' in self.ViTAS.ViTAS_model:
        #     img1_ = nn.functional.interpolate(img1, scale_factor=56/64, mode='bilinear', align_corners=True)
        #     img2_ = nn.functional.interpolate(img2, scale_factor=56/64, mode='bilinear', align_corners=True)
                
        features_out = self.ViTAS(img1,img2) # 
        feature1_list = features_out['feature0_out_list']
        feature2_list = features_out['feature1_out_list']
        
        feature1_list.reverse() 
        feature2_list.reverse()    
        pred_disp = self.igev(feature1_list, feature2_list, img1, img2, iters=iters, test_mode=test_mode)    
        # print(type(pred_disp))
        # print(self.training.shape)
        # print(pred_disp.shape)
          
        # if self.training
        # return pred_disp,corr_disp_list
        
        if test_mode:
            return pred_disp[-1]
        else: 
            return pred_disp
   