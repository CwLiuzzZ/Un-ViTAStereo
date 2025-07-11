import torch

from model_pack.dinoV2.dinov2.eval.setup import setup_and_build_model as dinoV2_model
from model_pack.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from YoYoModel.ViTASIGEV.ViTASIGEV import ViTASIGEVModel
# from YoYoModel.ViTASCre.ViTASCreModel import ViTASCreModel
from toolkit.args.model_args import get_dinov2_args_parser_1,dinoV2_config_dir_dic,dinoV2_ckpt_dir_dic


def prepare_model(hparams):
    if hparams.network == 'ViTASIGEV':
        model = load_ViTASIGEV_model(hparams)
    return model


def DepthAny2(encoder):
    assert encoder in ['vits','vitb','vitl']
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load('models/Depth_anything_v2/depth_anything_v2_{}.pth'.format(encoder), map_location='cpu',weights_only=True))
    depth_anything = depth_anything.cuda().eval()
    return depth_anything

def load_dinoV2_model(type='vitl'): # patch = 14
    assert type in ['vitl','vitb','vits','vitl_r','vitb_r','vits_r'] # l,b,s for size, r for register
    description = "Backbone for DINOv2"
    knn_args_parser = get_dinov2_args_parser_1(add_help=False)
    args = knn_args_parser.parse_args()
    # parents = [knn_args_parser]
    # args_parser = get_args_parser(description=description, parents=parents)
    # args = args_parser.parse_args()
    args.config_file = dinoV2_config_dir_dic[type]
    args.pretrained_weights = dinoV2_ckpt_dir_dic[type]
    model, autocast_dtype = dinoV2_model(args)
    # print(autocast_dtype)
    # model = ModelWithNormalize(model)
    return model

def load_ViTASIGEV_model(hparams): # yoyo's DiNOV2+CroCo
    model = ViTASIGEVModel(hparams.ViTAS_dic)
    return model
