import torch
import torch.nn as nn
from function.depth_function.local_ranking_loss import Local_Ranking_Loss,LoRa_Window_Vote_test
from toolkit.function.depth_function.depth_function import gradient_window,tensor_warp

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



def get_smooth_loss(disp, img,return_matrix=False,print_=False):
    """
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    if print_:
        print('grad_img_x nan check:',torch.isnan(grad_img_x).any())
        print('grad_disp_x nan check:',torch.isnan(grad_disp_x).any())
        print('grad_img_y nan check:',torch.isnan(grad_img_y).any())
        print('grad_disp_y nan check:',torch.isnan(grad_disp_y).any())

    if return_matrix:
        return grad_disp_x,grad_disp_y
    else:
        return grad_disp_x.mean() + grad_disp_y.mean()

def warp(img,disp,mode='right'):
    """
    img,disp in [B,H,W] or [B,C,H,W]
    mode in ['left','right']
    output in [B,C,H,W] with default_C=1
    """
    assert len(img.shape) in [3,4]
    assert len(disp.shape) in [3,4]
    if len(disp.shape) == 3:
        disp = disp.unsqueeze(1)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    assert img.shape[0] == disp.shape[0]
    _,_,H,W = img.shape
    assert disp.shape[-2] == H and disp.shape[-1] == W
    
    if mode == 'right':
        if torch.mean(disp)>0:
            disp=-disp
    elif mode == 'left':
        if torch.mean(disp)<0:
            disp=-disp
    else:
        raise ValueError
    disp = (disp/W).float()    
    batch_size, _, height, width = img.shape
    # generate grid
    x_base = torch.linspace(0, 1, W).repeat(batch_size,
                H, 1).type_as(img)
    y_base = torch.linspace(0, 1, H).repeat(batch_size,
                W, 1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = nn.functional.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)
    return output # [B,C,H,W]

# compute mean value on a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().cuda()
    return mean_value

class Main_Loss():
    def __init__(self,TopK=8):
        self.compute_ssim_loss = SSIM().cuda()
        self.local_ranking_loss = Local_Ranking_Loss(topK=TopK).cuda()
    
    def compute_pairwise_loss(self,imgL,imgR,l_disp,r_disp):
        imgR_warp = warp(imgR,l_disp) # warp right view with left disp
        r_disp_warp = warp(r_disp,l_disp) # warp right view with left disp

        diff_color = (imgL-imgR_warp).abs().mean(dim=1, keepdim=True)
        diff_disp = (l_disp-r_disp_warp).abs() / (l_disp+r_disp_warp)
        diff_img = (imgL-imgR_warp).abs().clamp(0, 1)
    
        # masking zero values
        valid_mask_ref = (imgR_warp.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (imgL.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask = valid_mask_tgt * valid_mask_ref    
        
        identity_warp_err = (imgL-imgR).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask
        ssim_map = self.compute_ssim_loss(imgL, imgR_warp)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        diff_img = torch.mean(diff_img, dim=1, keepdim=True)
        
        # reduce photometric loss weight for dynamic regions
        weight_mask = (1-diff_disp).detach()        
        diff_img = diff_img * weight_mask

        return diff_img, diff_disp, valid_mask

    #  r_disps_ast is flipped; depth_R is not flipped
    # input in [B,C,H，W], include the depths and disps
    def forward(self,l_disps, r_disps_ast, imgL, imgR, depth_L, depth_R, hparams):
        
        index = -1
        n_predictions = len(l_disps)
        loss_gamma = 0.9
        loss = 0
        
        for l_disp,r_disp_ast in zip(l_disps, r_disps_ast): 
            index+=1
            if hparams.weight_decay and n_predictions > 1:
                adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
                index_weight = adjusted_loss_gamma**(n_predictions - index - 1)
            else:
                index_weight = 1.0

            vote_r2l, vote_l2r, thr_l, thr_r,r2l_occ_mask,l2r_occ_mask = component_generate(l_disp,r_disp_ast,depth_L,depth_R,hparams.noc_ratio)
            
            diff_img_l, diff_disp_l, valid_mask_l = self.compute_pairwise_loss(imgL,imgR,l_disp,r_disp_ast.flip(-1)) # [B,1,H,W]          
            photo_loss_l = mean_on_mask(diff_img_l, valid_mask_l)
            geometry_loss_l = mean_on_mask(diff_disp_l, valid_mask_l)
                
            std_,mean_ = torch.std_mean(depth_L,dim=(1,2,3),keepdim=True)
            depth_L_ = (depth_L-mean_)/(std_+1e-8)  
            std__,mean__ = torch.std_mean(l_disp,dim=(1,2,3),keepdim=True)
            l_disp_ = (l_disp-mean__)/(std__+1e-8) 
            smooth_loss_l = (get_smooth_loss(l_disp, depth_L_.flip(-1))*5 + get_smooth_loss(depth_L.flip(-1), l_disp_)*1)/6
                        
            loss += index_weight*(photo_loss_l*hparams.photo_weight + geometry_loss_l*hparams.geometry_weight + smooth_loss_l*hparams.smooth_weight)
            
            if not hparams.LoRa_weight_l == 0:
                if not hparams.TopK ==0:
                    LoRa_loss_l = self.local_ranking_loss(l_disp,depth_L/(0.5*thr_l),vote_r2l)
                    loss += index_weight * hparams.LoRa_weight_l * LoRa_loss_l
            
            if not hparams.Depth_SSIM_weight_l == 0:
                Depth_SSIM_l = self.compute_ssim_loss(depth_L, l_disp).mean()
                loss += index_weight * hparams.Depth_SSIM_weight_l * Depth_SSIM_l
            
            if hparams.use_right:
                diff_img_r, diff_disp_r, valid_mask_r = self.compute_pairwise_loss(imgR.flip(-1),imgL.flip(-1),r_disp_ast,l_disp.flip(-1)) # [B,1,H,W]         
                photo_loss_r = mean_on_mask(diff_img_r, valid_mask_r)
                geometry_loss_r = mean_on_mask(diff_disp_r, valid_mask_r)
                
                std_,mean_ = torch.std_mean(depth_R,dim=(1,2,3),keepdim=True)
                depth_R_ = (depth_R-mean_)/(std_+1e-8)  
                std__,mean__ = torch.std_mean(r_disp_ast,dim=(1,2,3),keepdim=True)
                r_disp_ast_ = (r_disp_ast-mean__)/(std__+1e-8) 
                smooth_loss_r = (get_smooth_loss(r_disp_ast, depth_R_.flip(-1))*5 + get_smooth_loss(depth_R.flip(-1), r_disp_ast_)*1)/6
                                
                loss += index_weight*(photo_loss_r*hparams.photo_weight + geometry_loss_r*hparams.geometry_weight+ smooth_loss_r*hparams.smooth_weight) 
                                
                if not hparams.LoRa_weight_r == 0:
                    if not hparams.TopK ==0:
                        LoRa_loss_r = self.local_ranking_loss(r_disp_ast, depth_R.flip(-1)/(0.5*thr_r),vote_l2r)
                        loss += index_weight * hparams.LoRa_weight_r * LoRa_loss_r 
                
                if not hparams.Depth_SSIM_weight_r == 0:
                    Depth_SSIM_r = self.compute_ssim_loss(depth_R.flip(-1), r_disp_ast).mean()
                    loss += index_weight * hparams.Depth_SSIM_weight_r * Depth_SSIM_r 
                

            if torch.isnan(loss).any():
                
                print('photo_loss_l:',round(photo_loss_l.item(),3))
                print('geometry_loss_l:',round(geometry_loss_l.item(),3))
                print('smooth_loss_l:',round(smooth_loss_l.item(),3))
                print('LoRa_loss_l',round(LoRa_loss_l.item(),3))
                print('depth_SSIM_loss_l',round(Depth_SSIM_l.item(),3))
                print('photo_loss_r:',round(photo_loss_r.item(),3))
                print('geometry_loss_r:',round(geometry_loss_r.item(),3))
                print('smooth_loss_r:',round(smooth_loss_r.item(),3))
                print('LoRa_loss_r',round(LoRa_loss_r.item(),3))
                print('depth_SSIM_loss_r',round(Depth_SSIM_r.item(),3))

        return loss




def component_generate(disp_L,disp_R_ast,depth_L,depth_R,noc_ratio):
    disp_r2l = tensor_warp(disp_R_ast.flip(-1),disp_L) # [B,C,H,W]
    diff_r2l = torch.abs(disp_r2l-disp_L) # [B,C,H,W]
    r2l_mask = diff_r2l > torch.quantile(diff_r2l,noc_ratio) # True for unconfident disparities    
    disp_l2r = tensor_warp(disp_L.flip(-1),disp_R_ast) # [B,C,H,W]  
    diff_l2r = torch.abs(disp_l2r-disp_R_ast) # [B,C,H,W]  
    l2r_mask = diff_l2r > torch.quantile(diff_l2r,noc_ratio) # True for unconfident disparities  
    
    depth_gradient_l = gradient_window(depth_L,abs=False,rad=1)  # (B,C,H,W,N)
    disp_gradient_l = gradient_window(disp_L,abs=False,rad=1)  # (B,C,H,W,N)
    thr_l = torch.abs(depth_gradient_l).mean()/torch.abs(disp_gradient_l).mean()
    vote_r2l = LoRa_Window_Vote_test(disp_L, depth_L,thr = thr_l,return_score=True) # [B,C,H,W] high vote for unconfident disparities
    vote_r2l[r2l_mask] = 1
    depth_gradient_r = gradient_window(depth_R.flip(-1),abs=False,rad=1)  # (B,C,H,W,N)
    disp_gradient_r = gradient_window(disp_R_ast,abs=False,rad=1)  # (B,C,H,W,N)
    thr_r = torch.abs(depth_gradient_r).mean()/torch.abs(disp_gradient_r).mean()
    vote_l2r = LoRa_Window_Vote_test(disp_R_ast, depth_R.flip(-1),thr = thr_r,return_score=True)
    vote_l2r[l2r_mask] = 1
    
    return vote_r2l,vote_l2r,thr_l,thr_r,r2l_mask,l2r_mask

