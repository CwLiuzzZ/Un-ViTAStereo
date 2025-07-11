import torch
import torch.nn as nn
import torch.nn.functional as F
# import cv2
import yaml
import argparse
import numpy as np

# import sys
# sys.path.append('../../..')
from toolkit.function.base_function import seed_record

class D3_train_loss(nn.modules.Module):
    def __init__(self):
        super(D3_train_loss, self).__init__()
        self.SSIM_w = 0.85
        self.disp_gradient_w = 0.8
        self.disp_gradient_p = 0.5
        self.disp_gradient_s = 0.1
        # self.disp_gradient_h = 0.001
        self.lr_w = 0.8
        self.weight_overall()

        # for SSIM
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):

        disp = disp/disp.shape[-1]

        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        # x_shifts = disp[:, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + disp, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros',align_corners=True)

        return output

    def SSIM(self, x, y):

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, img):
        disp_gradients_x = self.gradient_x(disp.unsqueeze(1))
        disp_gradients_y = self.gradient_y(disp.unsqueeze(1))

        image_gradients_x = self.gradient_x(img) 
        image_gradients_y = self.gradient_y(img) 

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True)) 
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True)) 

        smoothness_x = torch.abs(disp_gradients_x) * weights_x
        smoothness_y = torch.abs(disp_gradients_y) * weights_y

        return smoothness_x + smoothness_y

    def DFM(self,sparse_gt,pre):
        mask = sparse_gt>0
        DFM_lss = torch.nn.functional.smooth_l1_loss(pre[mask],sparse_gt[mask])
        return DFM_lss

    def Heuristic(self,pre):
        return torch.mean(pre)

    def weight_overall(self):
        # sum = self.disp_gradient_w + self.disp_gradient_p + self.disp_gradient_s + self.disp_gradient_h
        sum = self.disp_gradient_w + self.disp_gradient_p + self.disp_gradient_s
        self.disp_gradient_w = self.disp_gradient_w / sum
        self.disp_gradient_p = self.disp_gradient_p / sum
        self.disp_gradient_s = self.disp_gradient_s / sum
        # self.disp_gradient_h = self.disp_gradient_h / sum

    def forward(self, left, right, disp_L, disp_R=None,sparse_disp=None):

        assert torch.min(left)  >= 0
        assert torch.min(right) >= 0

        # Generate images
        left_est = self.apply_disparity(right,-disp_L)
        # cv2.imwrite('generate_images/delete_left.png',left[0].permute(1,2,0).detach().cpu().numpy()*255)
        # cv2.imwrite('generate_images/delete_left_est.png',left_est[0].permute(1,2,0).detach().cpu().numpy()*255)
        # cv2.imwrite('generate_images/delete_right.png',right[0].permute(1,2,0).detach().cpu().numpy()*255)
        # disp_vis('generate_images/delete4.png',disp_L[0],128)
        # L1
        l1_left = torch.mean(torch.abs(left_est - left))
        # SSIM
        ssim_left = torch.mean(self.SSIM(left_est, left))
        image_loss_left = self.SSIM_w * ssim_left + (1 - self.SSIM_w) * l1_left
        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_L, left)
        disp_left_loss = torch.mean(disp_left_smoothness)

        if disp_R is None:
            if not sparse_disp is None:
                DFM_loss = self.DFM(sparse_disp,disp_L)
                loss = self.disp_gradient_w*DFM_loss+self.disp_gradient_p*image_loss_left+self.disp_gradient_s*disp_left_loss
                k = np.random.uniform(0, 1, 1)
                # if k > 0.95:
                #     print('DFM:',DFM_loss.item()*self.disp_gradient_w,'; ssim:',image_loss_left.item()*self.disp_gradient_s,'; SM:',disp_left_loss.item()*self.disp_gradient_w)
                #     print('loss:',loss.item())
            else:
                loss = self.disp_gradient_s*image_loss_left + self.disp_gradient_w * disp_left_loss
                k = np.random.uniform(0, 1, 1)
                # if k > 0.95:
                #     print('ssim:',image_loss_left.item(),'; SM:',disp_left_loss.item()*self.disp_gradient_w)
                #     print('loss:',loss.item())
            return loss
        else: # add Right disp cycle loss
            # Generate images
            right_est = self.apply_disparity(left, disp_R)
            # L1
            l1_right = torch.mean(torch.abs(right_est - right))
            # SSIM
            ssim_right = torch.mean(self.SSIM(right_est, right)) 
            image_loss_right = self.SSIM_w * ssim_right + (1 - self.SSIM_w) * l1_right
            image_loss = image_loss_left + image_loss_right
            # Disparities smoothness
            disp_right_smoothness = self.disp_smoothness(disp_R, right)
            disp_right_loss = torch.mean(torch.abs(disp_right_smoothness)) 
            disp_gradient_loss = disp_left_loss + disp_right_loss
            # L-R Consistency
            right_left_disp = self.apply_disparity(disp_R,-disp_L) 
            left_right_disp = self.apply_disparity(disp_L, disp_R) 
            lr_left_loss = [torch.mean(torch.abs(right_left_disp - disp_L))]
            lr_right_loss = [torch.mean(torch.abs(left_right_disp - disp_R)) ]
            lr_loss = lr_left_loss + lr_right_loss

            loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w*lr_loss
            return loss

def process_hparams_d3(hparams):
    if hparams.network == 'AANet':
        hparams.layer_size = [3,6,12]
        hparams.RBF_cycle = [9,15,15] # [9,15,15]
        hparams.ratio_th = [0.6,0.6,0.65]
        hparams.BF_i = 4
        hparams.D3_refine = True
        hparams.padding = 12
    elif hparams.network == 'BGNet':
        hparams.layer_size = [2,4,8]
        hparams.RBF_cycle = [11,15,15] # [11,15,15] # [1,5,1.82]
        # hparams.RBF_cycle = [1,5,9] # [11,15,15] # [1,5,1.82]
        hparams.ratio_th = [0.7,0.7,0.7]
        hparams.BF_i = 7
        hparams.D3_refine = True
        hparams.padding = 64
    else:
        raise ValueError('Invalid network type')
    return hparams


def sparse2map(points_A,points_B,img_H,img_W):
    mkpts0 = points_A.transpose(0,1)  # [n,2] numpy
    mkpts1 = points_B.transpose(0,1) 

    key_points_coordinate = torch.cat([mkpts0,mkpts1],dim=1)
    img1_seed = seed_record(img_W,img_H,key_points_coordinate) # [H,W]

    return img1_seed

class LaplacianLossBounded2(nn.Module): # used for CroCo-Stereo (except for ETH3D) ; in the equation of the paper, we have a=b
    def __init__(self, max_gtnorm=None, a=3.0, b=3.0):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = True
        self.a, self.b = a, b
        
    def forward(self, predictions, gt, conf):
        mask = gt > 0
        # mask[gt > 192]=False
        conf = 2 * self.a * (torch.sigmoid(conf / self.b) - 0.5 )
        return ( torch.abs(gt-predictions)[mask] / torch.exp(conf[mask]) + conf[mask] ).mean()# + torch.log(2) => which is a constant
    

def schedule_select(optimizer,hparams):
    if hparams.schedule == 'Cycle': # for large dataset
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.min_lr, max_lr=hparams.lr, step_size_up=int(hparams.epoch_steps/500 + 5), step_size_down=int(hparams.epoch_steps*0.4), cycle_momentum=False,mode='triangular2', last_epoch = hparams.this_epoch) # 20,6000;2,200
        print('lr_schedule Cycle: base_lr',hparams.min_lr,'; max_lr',hparams.lr)
    elif hparams.schedule == 'OneCycle': # for small dataset
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=hparams.lr,
            total_steps=hparams.num_steps + 50,
            pct_start=0.03,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch = hparams.this_epoch,
            # initial_lr = hparams.lr/25,
        )   
        print('lr_schedule OneCycle: max_lr',hparams.lr)
    lr_scheduler = {
        'scheduler': scheduler,
        'name': 'my_logging_lr',
        'interval':'step'
    }
    return lr_scheduler

def hparam_resume(hparams,evaluate = False):
    # if hparams.hparams_dir is None: # search for hparam file
    hparams.hparams_dir = '/'.join(hparams.ckpt_path.split('/')[:-1])+'/hparams.yaml'
    # print(hparams.hparams_dir)
    with open(hparams.hparams_dir, 'r') as f:
        hparams_ = yaml.unsafe_load(f)['hparams']
    
    # set exception
    if evaluate:
        exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','dataset','dataset_type','if_use_valid','val_dataset','val_dataset_type','save_name']
    else:
        if not hparams.resume_model:  # resume all
            exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','dataset','dataset_type','if_use_valid','val_dataset','val_dataset_type','save_name','batch_size','num_steps','epoch_steps','epoch_size','schedule']
        else: # only resume the model
            exception_ = ['resume','resume_model','ViTAS_dic','devices','ckpt_path','inference_type','num_workers','batch_size','num_steps','epoch_steps','epoch_size','schedule']
            
    # merge the hparams
    hparams_ = vars(hparams_)
    hparams = vars(hparams)    
    for k in hparams.keys():
        if k in hparams_.keys() and not k in exception_:
            hparams[k] = hparams_[k]
    hparams['ViTAS_dic'].update(hparams_['ViTAS_dic'])
    if not hparams['resume_model']: # resume all, continue training
        last_epoch = int(hparams['ckpt_path'].split('epoch=')[1].split('-')[0])
        hparams['this_epoch'] = last_epoch*hparams['epoch_steps']
        print('resume training from epoch {}, step {}'.format(last_epoch,hparams['this_epoch']))
    else: # resume model and re-start the training
        hparams['this_epoch'] = -1
    return argparse.Namespace(**hparams)