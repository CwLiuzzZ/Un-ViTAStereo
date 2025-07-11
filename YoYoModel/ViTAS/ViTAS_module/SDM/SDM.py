import torch.nn as nn
import torch
# from YoYoModel.ViTAS.ViTAS_module.SDM.featup.JBU import JBUStack


def forward_preprocess_general(preprocesses,VFM_features):
    feature0_list = []
    feature1_list = []
    features_list = []
    for i in range(len(VFM_features)):
        features_list.append(preprocesses[i](VFM_features[i])) # from high_res to low_res
    for i in range(len(VFM_features)):
        features = features_list[i]
        chunk = torch.chunk(features, chunks=2, dim=0)
        feature0_list.append(chunk[0])
        feature1_list.append(chunk[1]) 
    return feature0_list,feature1_list
    
# def forward_preprocess_featup(JBU,img_concat,VFM_features,):
#     features_list = []
#     feature0_list = []
#     feature1_list = []
        
#     features_list.append(JBU.act_1_preprocess(VFM_features[0],img_concat))
#     features_list.append(JBU.act_2_preprocess(VFM_features[1],img_concat))
#     features_list.append(JBU.act_3_preprocess(VFM_features[2]))
#     features_list.append(JBU.act_4_preprocess(VFM_features[3]))
#     for i in range(len(VFM_features)):
#         features = features_list[i]
#         chunk = torch.chunk(features, chunks=2, dim=0)
#         feature0_list.append(chunk[0])
#         feature1_list.append(chunk[1])
#     return feature0_list,feature1_list # [32,64,128,256]
    
def make_preprocess(ViTAS_channel=1024,out_channels=[48,64,192,160],midd_channels=[48,64,192,160],if_featup=False):  
    
    if if_featup:
        return JBUStack(ViTAS_channel,out_channels,midd_channels)
        
    elif not if_featup:   
        act_1_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=midd_channels[0],
                out_channels=out_channels[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        act_2_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=midd_channels[1],
                out_channels=out_channels[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        act_3_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        act_4_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=midd_channels[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=midd_channels[3],
                out_channels=out_channels[3],
                kernel_size=3, stride=2, padding=1,
            )
        )


        return  nn.ModuleList([
                act_1_preprocess, # 1/4
                act_2_preprocess, # 1/8
                act_3_preprocess, # 1/16
                act_4_preprocess  # 1/32
            ])
    else:
        raise ValueError