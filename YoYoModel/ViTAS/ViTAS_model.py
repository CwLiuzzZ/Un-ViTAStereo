import torch
import torch.nn as nn
# import time
# import numpy as np

from toolkit.args.model_args import get_dinov2_args_parser_1,dinoV2_config_dir_dic,dinoV2_ckpt_dir_dic
from model_pack.dinoV2.dinov2.eval.setup import setup_and_build_model as dinoV2_model
from YoYoModel.ViTAS.ViTAS_module.Uni_CA.CA_transformer import FeatureTransformer
from YoYoModel.ViTAS.ViTAS_module.Uni_CA.CA_utils import feature_add_position
# from mmseg.models.backbones.beit import BEiT
# from model_pack.Depth_Anything.depth_anything.dpt import DepthAnything_body

from YoYoModel.ViTAS.ViTAS_module.SDFA_fuse.SDFA import SDFA
from YoYoModel.ViTAS.ViTAS_module.PAFM_fuse.PAFM import PAFM
from YoYoModel.ViTAS.ViTAS_module.VFM_fuse.VFM import VFM
from YoYoModel.ViTAS.ViTAS_module.SDM.SDM import make_preprocess,forward_preprocess_general # ,forward_preprocess_featup

def get_parameter_number(model,name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, ' Total:', total_num, 'Trainable:', trainable_num)
    # return {'Total': total_num, 'Trainable': trainable_num}
    
class ViTASBaseModel(nn.Module):
    def __init__(self,VFM_type='DINOv2',ViTAS_hooks=[5,11,17,23],ViTAS_channel=1024,pre_channels=[64,128,256,512], out_channels=[48,64,192,160],attn_splits_list=[8,4,2,2],CA_layers=2,wo_fuse=False,ViTAS_model='vit_l',ViTAS_unfreeze='0',ViTAS_pure=False,ViTAS_fuse='PAFM',fuse_dic = {},block_expansion=0,featup_SDM=False):
        # CA 2760128 with 2 layers
        # SDM 3033040 with 4 blocks
        # PAFM 224908 with LA and GA
        # dinov2 204368640: tranable: 62993408 of 3 unfrozen
        super().__init__()
        print('VFM_type:',VFM_type)
        print('ViTAS_model:',ViTAS_model)
        print('ViTAS_hooks:',ViTAS_hooks)
        print('ViTAS_unfreeze:',ViTAS_unfreeze)
        # print('ViTAS_wo_fuse:',wo_fuse)
        print('CA_layers:',CA_layers)
        print('featup_SDM',featup_SDM)
        print('block_expansion',block_expansion)
        print('ViTAS_fuse:',ViTAS_fuse, ' fuse_dic:',fuse_dic)
        assert len(ViTAS_hooks)==len(pre_channels)==len(out_channels)==len(attn_splits_list),str(len(ViTAS_hooks))+' '+str(len(pre_channels))+' '+str(len(out_channels))+' '+str(len(attn_splits_list))
        # Dino config
        self.VFM_type = VFM_type
        self.ViTAS_model = ViTAS_model  
        self.ViTAS_hooks = ViTAS_hooks  
        self.ViTAS_fuse = ViTAS_fuse   
        self.CA_layers = CA_layers   
        self.featup_SDM = featup_SDM   
        self.time_list = []
        self.pre_midd_channel = None
        self.VFM  = load_vfm(VFM_type,ViTAS_model,ViTAS_unfreeze,block_expansion)
        
        if not ViTAS_pure:
            # Adapter config
            self.attn_splits_list = attn_splits_list # for CA
            
            # print('ViTAS_channel',ViTAS_channel)
            # print('pre_channels',pre_channels)
            # print('out_channels',out_channels)
            
            self.pre_channels,self.out_channels,self.pre_midd_channel= channels_init(self.ViTAS_fuse,ViTAS_channel,pre_channels,out_channels)
            
            # print('pre_channels',self.pre_channels)
            # print('out_channels',self.out_channels)
            # print('pre_channels',self.pre_midd_channel)
            # exit()
            
            
            self.wo_fuse = wo_fuse
            if self.wo_fuse: # without fuse
                self.pre_channels = out_channels
            else: # with fuse
                if self.ViTAS_fuse == 'SDFA':
                    self.fuse_blocks = SDFA(in_channels = self.pre_channels, out_channels=self.out_channels)
                if self.ViTAS_fuse == 'PAFM':
                    self.fuse_blocks = PAFM(self.pre_channels,**fuse_dic)
                if self.ViTAS_fuse == 'VFM':
                    self.fuse_blocks = VFM(self.pre_channels,**fuse_dic)
                get_parameter_number(self.fuse_blocks,'fuse')
            # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.preprocess = make_preprocess(ViTAS_channel=ViTAS_channel,out_channels=self.pre_channels,midd_channels = self.pre_midd_channel,if_featup=featup_SDM)
            self.CA = make_CA(CA_layers=self.CA_layers, channels=self.out_channels)
            get_parameter_number(self.VFM,VFM)
            get_parameter_number(self.preprocess,'SDM')
            get_parameter_number(self.CA,'CA')
        
    
    def forward(self,img1_ori, img2_ori,get_before_CA=False,get_pre_process=False): 
        
        features_out = {'feature0_out_list':[],'feature1_out_list':[],'feature0_before_CA_list':[],'feature1_before_CA_list':[],'feature0_resize_list':[],'feature1_resize_list':[]}
        if self.VFM_type in ['DINOv2','DepthAny','DepthAny2']:
            img1 = nn.functional.interpolate(img1_ori, scale_factor=7/8, mode='bilinear', align_corners=True)
            img2 = nn.functional.interpolate(img2_ori, scale_factor=7/8, mode='bilinear', align_corners=True)
        
        image_concat = torch.cat((img1, img2), dim=0)  # [2B, C, H, W]
        image_ori_concat = torch.cat((img1_ori, img2_ori), dim=0)  # [2B, C, H, W]
        # Dino_features = self.vfm.get_intermediate_layers(image_concat,n=self.ViTAS_hooks,reshape=True) # from high_res to low_res
        VFM_features = self.get_intermediate_layers(image_concat,n=self.ViTAS_hooks,reshape=True) # from high_res to low_res
        
        len_ = len(VFM_features)
        time_list = []
        
        
        # if not self.featup_SDM:
        feature0_list,feature1_list = forward_preprocess_general(self.preprocess,VFM_features) 
        # else:
        #     feature0_list,feature1_list = forward_preprocess_featup(self.preprocess,image_ori_concat,VFM_features)
        
        # feature0_list = []
        # feature1_list = []        
        # features_list = []
    
        # features_list = forward_preprocess(VFM_features,if_featup=self.featup_SDM)
        # features_list = []
        # for i in range(len_):
        #     features_list.append(self.preprocess[i](VFM_features[i])) # from high_res to low_res
        # for i in range(len_):
        #     features = features_list[i]
        #     chunk = torch.chunk(features, chunks=2, dim=0)
        #     feature0_list.append(chunk[0])
        #     feature1_list.append(chunk[1]) 
        
        if get_pre_process:
            for i in feature1_list:
                features_out['feature0_resize_list'].append(i)
                features_out['feature1_resize_list'].append(i)
        
        
        x_l = feature0_list[-1] # left , lowest_res
        x_r = feature1_list[-1] # right, lowest_res        
        for i in range(len_-1, -1, -1): # i = 3,2,1,0, fuse from low_res to high_res    
            if get_before_CA:  
                features_out['feature0_before_CA_list'].append(x_l)
                features_out['feature1_before_CA_list'].append(x_r)
            # add PE and cross-attention Transformer
            if not self.CA_layers == 0:
                # print(x_l.shape)
                x_l, x_r = feature_add_position(x_l, x_r, self.attn_splits_list[i], self.out_channels[i])
                # print(x_l.shape)
                x_l, x_r = self.CA[i](x_l, x_r,attn_type='self_swin2d_cross_swin1d',attn_num_splits=self.attn_splits_list[i])
                # print(x_l.shape)
                # print('----------')
                        
            features_out['feature0_out_list'].append(x_l)
            features_out['feature1_out_list'].append(x_r)
            if i > 0:
                if self.wo_fuse: # without fuse
                    x_l = feature0_list[i-1]
                    x_r = feature1_list[i-1]
                else: # with fuse
                    # time0 = time.time()
                    x_l,x_r = self.fuse_blocks.fuse(i,x_l,x_r,feature0_list[i-1],feature1_list[i-1])
                    # time_list.append(time.time()-time0)
        # 
        # self.time_list.append(np.sum(time_list))
        # if len(self.time_list) > 40:
            # print('fuse time:', np.mean(self.time_list[20:]))
        
        # exit()
        
        return features_out # return lists of features from low resolution to high resolution

    def get_intermediate_layers(self,img,n,reshape):
        # if self.VFM_type in ['DINOv2','BEiT2','DepthAny','DepthAny2']:
        return self.VFM.get_intermediate_layers(img,n,reshape)
        # else:
            # raise ValueError("{}, Wrong VFM type is given.".format(self.VFM_type))
    def forward_patch_features(self,img):
        # if self.VFM_type in ['DINOv2','BEiT2','DepthAny','DepthAny2']:
        return self.VFM.forward_patch_features(img)
        # else:
        #     raise ValueError("{}, Wrong VFM type is given.".format(self.VFM_type))

def load_vfm(VFM_type,ViTAS_model,ViTAS_unfreeze,block_expansion):
    if VFM_type == 'DINOv2':
        knn_args_parser = get_dinov2_args_parser_1(add_help=False)
        args = knn_args_parser.parse_args()
        args.block_expansion = block_expansion
        args.config_file = dinoV2_config_dir_dic[ViTAS_model]
        args.pretrained_weights = dinoV2_ckpt_dir_dic[ViTAS_model]
        model, autocast_dtype = dinoV2_model(args)
        model = DINOv2_weights_unfreeze(model,int(ViTAS_unfreeze)+block_expansion)
    # elif VFM_type == 'BEiT2':
    #     model = BEiT(
    #         embed_dims=1024,
    #         num_layers=24,
    #         num_heads=16,
    #         mlp_ratio=4,
    #         qv_bias=True,
    #         init_values=1e-6,
    #         drop_path_rate=0.2,
    #         # out_indices=[7, 11, 15, 23],
    #         init_cfg = dict(type='Pretrained', checkpoint='../toolkit/models/BEiT2/mmseg_beit2.pth'))
    #     model.init_weights()
    #     model = BEiT_weights_unfreeze(model,ViTAS_unfreeze)
    elif VFM_type == 'DepthAny':
        knn_args_parser = get_dinov2_args_parser_1(add_help=False)
        args = knn_args_parser.parse_args()
        args.block_expansion = block_expansion
        args.config_file = '../model_pack/dinoV2/dinov2/configs/eval/vitl14_pretrain.yaml'
        args.pretrained_weights = "../toolkit/models/depth_anything/DepthAny_body_vitl.pth"
        model, autocast_dtype = dinoV2_model(args)
        model = DINOv2_weights_unfreeze(model,ViTAS_unfreeze+block_expansion)
    elif VFM_type == 'DepthAny2':
        knn_args_parser = get_dinov2_args_parser_1(add_help=False)
        args = knn_args_parser.parse_args()
        args.block_expansion = block_expansion
        args.config_file = '../model_pack/dinoV2/dinov2/configs/eval/vitl14_pretrain.yaml'
        args.pretrained_weights = "../toolkit/models/Depth_anything_v2/DepthAny2_body_vitl.pth"
        model, autocast_dtype = dinoV2_model(args)
        model = DINOv2_weights_unfreeze(model,int(ViTAS_unfreeze)+block_expansion)
    else:
        raise ValueError("Wrong VFM type is given.")
    return model 



def channels_init(fuse_mode,ViTAS_channel,pre_channels,out_channels):
    if fuse_mode == 'SDFA':
        # pre_channels[-1] = out_channels[-1]
        # pre_midd_channel = pre_channels
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    elif 'PAFM' in fuse_mode:
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    elif 'VFM' in fuse_mode:
        pre_channels = out_channels
        pre_midd_channel = [int(ViTAS_channel/2)]*4
    else:
        raise ValueError('Unknown fuse mode :{}'.format(fuse_mode))
    return pre_channels,out_channels,pre_midd_channel

    
def make_CA(CA_layers=2,channels=[48,64,192,160],ffn_dim_expansion=[2,2,2,2]):      
    CA_1 = FeatureTransformer(num_layers=CA_layers,d_model=channels[0],nhead=1,ffn_dim_expansion=ffn_dim_expansion[0])
    CA_2 = FeatureTransformer(num_layers=CA_layers,d_model=channels[1],nhead=1,ffn_dim_expansion=ffn_dim_expansion[1])
    CA_3 = FeatureTransformer(num_layers=CA_layers,d_model=channels[2],nhead=1,ffn_dim_expansion=ffn_dim_expansion[2])
    CA_4 = FeatureTransformer(num_layers=CA_layers,d_model=channels[3],nhead=1,ffn_dim_expansion=ffn_dim_expansion[3])
    return  nn.ModuleList([CA_1,CA_2,CA_3,CA_4])

# def make_fuse()


def interpolate_features(feature_list):
    output = []
    output.append(torch.nn.functional.interpolate(feature_list[0], scale_factor=4, mode='bilinear', align_corners=True))
    output.append(torch.nn.functional.interpolate(feature_list[1], scale_factor=2., mode='bilinear', align_corners=True))
    output.append(feature_list[2])
    output.append(torch.nn.functional.interpolate(feature_list[3], scale_factor=0.5, mode='bilinear', align_corners=True))
    return  output

# def BEiT_weights_unfreeze(model,ViTAS_unfreeze):
#     if ViTAS_unfreeze=='0':
#         for p in model.parameters():
#             p.requires_grad = False
#     elif ViTAS_unfreeze=='1':
#         for k,v in model.named_parameters():
#             if not 'layers.23' in k:
#                 v.requires_grad=False 
#     elif ViTAS_unfreeze=='2':
#         for k,v in model.named_parameters():
#             if not ('layers.22' in k or 'layers.23' in k):
#                 v.requires_grad=False 
#     elif ViTAS_unfreeze=='3':
#         for k,v in model.named_parameters():
#             if not ('layers.21' in k or 'layers.22' in k or 'layers.23' in k):
#                 v.requires_grad=False 
#     elif ViTAS_unfreeze=='4':
#         for k,v in model.named_parameters():
#             if not ('layers.20' in k or 'layers.21' in k or 'layers.22' in k or 'layers.23' in k):
#                 v.requires_grad=False 
#     elif ViTAS_unfreeze=='5':
#         for k,v in model.named_parameters():
#             if not ('layers.19' in k or 'layers.20' in k or 'layers.21' in k or 'layers.22' in k or 'layers.23' in k):
#                 v.requires_grad=False 
#     return model

def DINOv2_weights_unfreeze(model,ViTAS_unfreeze):
    
    total_blocks = model.n_blocks
    unfreeze_keys_list = ['norm.weight','norm.bias']
    for i in range(total_blocks-1,total_blocks-1-ViTAS_unfreeze,-1):
        unfreeze_keys_list.append('blocks.{}'.format(str(i)))
    
    for k,v in model.named_parameters():
        check = [i in k for i in unfreeze_keys_list]
        if any(check):
            v.requires_grad=True
        else:
            v.requires_grad=False
    return model



# def make_preprocess(ViTAS_channel=1024,out_channels=[48,64,192,160],midd_channels=[48,64,192,160]):     
#     act_1_preprocess = nn.Sequential(
#         nn.Conv2d(
#             in_channels=ViTAS_channel,
#             out_channels=midd_channels[0],
#             kernel_size=1, stride=1, padding=0,
#         ),
#         nn.ConvTranspose2d(
#             in_channels=midd_channels[0],
#             out_channels=out_channels[0],
#             kernel_size=4, stride=4, padding=0,
#             bias=True, dilation=1, groups=1,
#         )
#     )

#     act_2_preprocess = nn.Sequential(
#         nn.Conv2d(
#             in_channels=ViTAS_channel,
#             out_channels=midd_channels[1],
#             kernel_size=1, stride=1, padding=0,
#         ),
#         nn.ConvTranspose2d(
#             in_channels=midd_channels[1],
#             out_channels=out_channels[1],
#             kernel_size=2, stride=2, padding=0,
#             bias=True, dilation=1, groups=1,
#         )
#     )

#     act_3_preprocess = nn.Sequential(
#         nn.Conv2d(
#             in_channels=ViTAS_channel,
#             out_channels=out_channels[2],
#             kernel_size=1, stride=1, padding=0,
#         )
#     )

#     act_4_preprocess = nn.Sequential(
#         nn.Conv2d(
#             in_channels=ViTAS_channel,
#             out_channels=midd_channels[3],
#             kernel_size=1, stride=1, padding=0,
#         ),
#         nn.Conv2d(
#             in_channels=midd_channels[3],
#             out_channels=out_channels[3],
#             kernel_size=3, stride=2, padding=1,
#         )
#     )


#     return  nn.ModuleList([
#             act_1_preprocess, # 1/4
#             act_2_preprocess, # 1/8
#             act_3_preprocess, # 1/16
#             act_4_preprocess  # 1/32
#         ])