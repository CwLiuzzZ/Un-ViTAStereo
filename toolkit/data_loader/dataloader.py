import numpy as np
# from PIL import Image
import math
from torch.utils.data import Dataset
import cv2
import os

# import sys
# sys.path.append('../..')
from toolkit.function.base_function import io_disp_read
from toolkit.data_loader.transforms import Augmentor 

aug_config_dic_train = {'ViTASIGEV':{'RandomColor':True,'VFlip':False,'crop':(320,700),'rotate':False,'scale':True,'erase':True,'color_diff':False},
                        }
aug_config_dic_evaluate = {'ViTASIGEV':{'crop':None}, 
                        'Depth':{'norm':True,'crop':None}, # for DepthAnythingV2
                        }


def dataloader_customization(hparams):
    network = hparams.network

    #########################
    ### adjust aug_config ### 
    #########################
    
    if 'train' in hparams.inference_type:
        aug_config = aug_config_dic_train[network].copy()
    elif 'evaluate' in hparams.inference_type:
        aug_config = aug_config_dic_evaluate[network].copy()
    valid_aug_config = aug_config_dic_evaluate[network].copy()
    
    if hparams.keep_size:
        aug_config['resize']=None

    if 'KITTI' in hparams.val_dataset:
        if hparams.batch_size>1:
            if valid_aug_config['crop'] is None:
                valid_aug_config['crop']=(370,1224)

    if not hparams.resize is None:
        aug_config['resize']=hparams.resize
    
    # 'scale can be deployed only when crop_size is set'
    if aug_config['crop'] is None:
        if 'scale' in aug_config.keys():
            assert not aug_config['scale'], 'scale can be deployed only when crop_size is set'

    if 'idd' in hparams.dataset:
        hparams.max_disp = None
    
    
    return hparams,aug_config,valid_aug_config

base_lr_dic = {'ViTASIGEV':{'lr':8e-5,'min_lr':1e-5}, # 7~0.8
               }

max_disp_dic = {'ViTASIGEV':192,
               }

def optimizer_customization(hparams,n_img):
    # return hparams
    # print(n_img,hparams.batch_size,hparams.devices)
    hparams.epoch_steps = math.ceil(n_img/(hparams.batch_size*len(hparams.devices)))
    hparams.num_steps = hparams.epoch_steps*hparams.epoch_size
    print('total steps:', hparams.num_steps, ' epoch steps:', hparams.epoch_steps,' total epoch: ',hparams.epoch_size)
    # exit()
    if hparams.num_steps > 300000: # 50000
        hparams.schedule = 'Cycle' # for large dataset
    else:
        hparams.schedule = 'OneCycle' # for small dataset
    if hparams.network in base_lr_dic.keys():
        hparams.lr = base_lr_dic[hparams.network]['lr']
        hparams.min_lr = base_lr_dic[hparams.network]['min_lr'] 
    else:
        hparams.lr = 1e-4
        hparams.min_lr = 1e-5
    if hparams.network in max_disp_dic.keys():
        hparams.max_disp = max_disp_dic[hparams.network]
    else:
        hparams.max_disp = 192
    return hparams

def prepare_dataset(file_paths_dic, aug_config,inference_type,save_method):

    '''
    function: make dataloader
    input:
        file_paths_dic: store file paths
        aug_config: configuration for augment
    output:
        dataloader
    '''
    # augmentation
    transformer = Augmentor(**aug_config)
    
    dataset = DepthDataset(file_paths_dic,transform=transformer,save_method=save_method)
    
    n_img = len(dataset)
    print('Use a dataset with {} image pairs'.format(n_img))
    return dataset,n_img

def default_loader(path):
    '''
        function: read left and right images
        output: array
    '''
    # return Image.open(path).convert('RGB')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class DepthDataset(Dataset):
    def __init__(self, file_paths_dic,transform, loader=default_loader, dploader=io_disp_read,save_method=None):
        super(DepthDataset, self).__init__()
        self.transform = transform
        self.loader = loader
        self.disploader = dploader
        self.samples = []
        self.load_disp = True
        self.save_method = save_method

        self.lefts = file_paths_dic['left_list']
        self.rights = file_paths_dic['right_list']
        self.disps = file_paths_dic['disp_list']
        self.save_dirs1 = file_paths_dic['save_path_disp']
        self.save_dirs2 = file_paths_dic['save_path_disp_image']
        self.calib = file_paths_dic['calib_list'] 
        
        # print('number of files: left image {}, right image {}, disp {}'.format(len(self.lefts), len(self.rights), len(self.disps)))
        assert len(self.lefts) == len(self.rights), "{},{}".format(len(self.lefts),len(self.rights))
        if not len(self.disps) == len(self.lefts):
            print('warning: disp file numbers not equal image pair numbers, use zero disparity map')
            self.load_disp = False
        # assert len(self.lefts) == len(self.rights) == len(self.disps), "{},{},{}".format(len(self.lefts),len(self.rights),len(self.disps))
        for i in range(len(self.lefts)):
            sample = dict()
            sample['left'] = self.lefts[i]
            sample['right'] = self.rights[i]
            # sample['calib'] = self.calib[i]
            # assert not sample['calib'] is None
            if self.load_disp:
                sample['disp'] = self.disps[i]
            sample['save_dir1'] = self.save_dirs1[i]
            sample['save_dir2'] = self.save_dirs2[i]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        sample = {}
        sample_path = self.samples[index]
    
    
        sample['left'] = self.loader(sample_path['left']) # array
        sample['right'] = self.loader(sample_path['right']) # array
        if os.path.exists(sample_path['disp']):
            sample['disp'] = self.disploader(sample_path['disp'])
        else:
            sample['disp'] = np.zeros(shape=(np.array(sample['left']).shape[0],np.array(sample['left']).shape[1]))
        sample['disp_dir'] = sample_path['disp']
        sample['left_dir'] = sample_path['left']
        sample['right_dir'] = sample_path['right']
        sample['save_dir_disp'] = sample_path['save_dir1']
        sample['save_dir_disp_vis'] = sample_path['save_dir2']        
        
        sample['append'] = None
        left_depth_path = sample['save_dir_disp'].replace(self.save_method,'DepthAnything').replace('.npy','_depthL.npy')
        right_depth_path = sample['save_dir_disp'].replace(self.save_method,'DepthAnything').replace('.npy','_depthR.npy')
        
        if os.path.exists(left_depth_path) and os.path.exists(right_depth_path):
            sample['append'] = {}
            sample['append']['left_depth'] = np.load(left_depth_path)
            sample['append']['right_depth'] = np.load(right_depth_path)
            
        # for i in sample:
        #     print(i,type(sample[i]))  
        sample = self.transform(sample)
        
        # for i in sample:
        #     print(i,type(sample[i]))  
        if sample['append'] is None:
            sample['append'] = 'None'
        
        # for i in sample:
        #     print(i,type(sample[i]))    
        # exit()
        return sample