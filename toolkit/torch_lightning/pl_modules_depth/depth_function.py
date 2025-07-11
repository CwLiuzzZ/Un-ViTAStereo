import torch
import numpy as np
import cv2
from function.base_function import disp_vis
from function.D3_function import FBS_init,dense_feature_matching,draw_points,rescale_points,sim_construct
from model_pack.D3Stereo.seed_growth import seed_growth,subpixel_enhancement
import os
import torch.nn.functional as F

def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        vgrid = grid.cuda()
    else:
        vgrid = grid
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

def SSIM( x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def gradient( pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def compute_grad2_smoothness_loss(flo, image, beta=1.0):
    """
    Calculate the image-edge-aware second-order smoothness loss
    """
    
    img_grad_x, img_grad_y = gradient(image)
    # weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    # weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))
    weights_x = torch.exp(torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    mean_flo = flo.mean(2, True).mean(3, True)
    norm_flo = flo / (mean_flo + 1e-7)
    flo = norm_flo

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

def reconstruction_loss( x, y):
    ssim = torch.mean(SSIM(x, y))
    l1 = torch.mean(torch.abs(x - y))
    return 0.85*ssim + 0.15*l1

def compute_loss(l_disps, r_disps_ast, imgL, imgR, hparams):
    
    imgL_ast = imgL.flip(3)

    loss = 0
        
    for l_disp,r_disp_ast in zip(l_disps, r_disps_ast):        
        # compute left reconstruction loss
        recon_imgL = warp(imgR, l_disp)
        left_recon_loss = reconstruction_loss(
                                    recon_imgL[:,:,:, 75:575], imgL[:,:,:, 75:575])
        loss += left_recon_loss
        # print('loss0_recon:',left_recon_loss)
        
        
        # print('left_recon_loss:',round(left_recon_loss.item(),3))
        
        if hparams.use_right:
            r_disp = r_disp_ast.flip(3)
            # compte right reconstruction loss
            recon_imgR_ast = warp(imgL_ast, r_disp_ast)
            recon_imgR = recon_imgR_ast.flip(3)
            right_recon_loss = reconstruction_loss(
                                            recon_imgR[:,:,:, 0:500], imgR[:,:,:, 0:500])
            loss += right_recon_loss

        
        # compute total loss
        if not hparams.smooth_weight == 0:
            
            left_smth_loss = compute_grad2_smoothness_loss(l_disp, imgL) # disp/20
            loss += hparams.smooth_weight * left_smth_loss  
        
            if hparams.use_right:
                # compute right disparity smoothness loss
                right_smth_loss = compute_grad2_smoothness_loss(r_disp, imgR) # disp/20
                loss += hparams.smooth_weight * right_smth_loss 

        
    return loss
