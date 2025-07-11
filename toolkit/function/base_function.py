import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import re
# from pylab import math
import math
import sys
import skimage.io
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from torch.linalg import det

def single_image_warp(img,disp,mode='right', tensor_output = False):
    '''
    function:
        warp single image with disparity map to another perspective
    input:
        img: image; should be 2D or 3D array
        disp: disparity map; should be 2D array or tensor
        mode: perspective of the input image
    output:
    '''
    assert mode in ['left','right']

    if not isinstance(img, torch.Tensor):
        if len(img.shape)==3:
            img = np.transpose(img,(2,0,1))
        img = img.astype(np.float32)
        img = torch.from_numpy(img) # [C,H,W] or [H,W]

    if not isinstance(disp, torch.Tensor):
        disp = torch.tensor(disp)
    disp = (disp/disp.shape[1]).float()

    if mode == 'left':
        # should be negative disparity
        if torch.mean(disp)<0:
            disp=-disp
    elif mode == 'right':
        # should be positive disparity
        if torch.mean(disp)>0:
            disp=-disp
    
    # disp = torch.from_numpy(disp/disp.shape[1]).float()
    assert img.shape[-1] == disp.shape[-1], str(img.shape)+' and '+str(disp.shape)
    assert img.shape[-2] == disp.shape[-2], str(img.shape)+' and '+str(disp.shape)

    disp = disp.unsqueeze(0).unsqueeze(0)
    if len(img.shape)==2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape)==3:
        img = img.unsqueeze(0)
        
    batch_size, _, height, width = img.shape
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)

    if tensor_output: 
        return output # [1,C,H,W]

    if output.shape[1]==1:
        output = output[0][0].detach().numpy()
    else:
        output = output[0].detach().numpy()
    if len(output.shape)==3:
        output = np.transpose(output,(1,2,0))

    return output

# COLORMAP_JET, COLORMAP_PARULA, COLORMAP_MAGMA, COLORMAP_PLASMA, COLORMAP_VIRIDIS
# disp should be [H,W] numpy.array
def disp_vis(save_dir,disp,max_disp=None,min_disp=None,colormap=cv2.COLORMAP_JET,inverse=False):
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if min_disp is None:
        min_disp = np.min(disp[:-3,:-3])
    if max_disp is None:
        max_disp = np.max(disp[:-3,:-3])
    else:
        save_dir_split = save_dir.split('.png')
        save_dir = save_dir_split[0]+'_{}_{}'.format(str(round(max_disp,1)),str(round(min_disp,1)))+'.png'
    # max_disp = (int(max_disp/5)+1)*5
    # min_disp = int(min_disp/5)*5
    disp = np.clip(disp,min_disp,max_disp)
    disp = 255 * (disp-min_disp)/(max_disp-min_disp)
    # disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)        
    disp=disp.astype(np.uint8)
    if inverse:
        disp = 255 - disp
    disp = cv2.applyColorMap(disp,colormap) # cv2.COLORMAP_JET
    
    if not save_dir is None:
        cv2.imwrite(save_dir,disp)
    else:
        return disp

# COLORMAP_JET, COLORMAP_PARULA, COLORMAP_MAGMA, COLORMAP_PLASMA, COLORMAP_VIRIDIS
# disp should be [H,W] numpy.array
def disp_D1_vis(save_dir,disp,max_disp=None,min_disp=0,colormap=cv2.COLORMAP_JET,inverse=False,max_color=255,min_color=0,backupcolor=0,error_color=255):
    
    assert len(disp.shape) == 2, len(disp.shape)
    if torch.is_tensor(disp):
        disp = disp.detach().cpu().numpy()
    if max_disp is None:
        max_disp = np.mean(disp)

    mask1 = disp>max_disp
    mask2 = disp==0

    disp[~mask1] = min_color+(max_color-min_color) * (disp[~mask1]-min_disp)/(max_disp-min_disp)
    disp[mask1] = error_color
    disp[mask2]=backupcolor 
    disp=disp.astype(np.uint8)
    if inverse:
        disp = 255 - disp
    disp = cv2.applyColorMap(disp,colormap)
    # print(disp.shape)
    # disp[mask1,:] = [40, 100, 255]
    # disp[mask1,:] = [74, 74, 169]
    # disp[mask1,:] = [74, 28, 125]
    # disp[mask1,:] = [35, 157, 255]
    disp[mask1,:] = [40, 110, 255]
    disp[mask2,:]=[255,255,255] 
    cv2.imwrite(save_dir,disp)


def dirs_walk(dir_list):
    '''
    output:
        all the files in dir_list
    '''
    file_list = []
    for dir in dir_list:
        paths = os.walk(dir)
        for path, dir_lst, file_lst in paths:
            file_lst.sort()
            for file_name in file_lst:
                file_path = os.path.join(path, file_name)
                file_list.append(file_path)
    file_list.sort()
    return file_list 


# input img: cv2.imread() numpy
def get_disp(imgL, imgR, method,max_disp=None):

    '''
    fuction: generate disparity map with SGBM or BM algorithm
    input:
        imgL: left image; should be a 3D array
        imgR: right image; should be a 3D array
        method: method   
    '''
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    min_disp = 0
    # SGBM Parameters: wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    window_size = 5
    if not max_disp is None:
        max_disp=255

    max_disp = 36

    if method == 'SGBM':    
        img_channel = 3
        imgL=cv2.imread(imgL)
        imgR=cv2.imread(imgR)
        # imgL_flip = np.flip(imgL,1)
        # imgR_flip = np.flip(imgR,1)
        # print(imgL.shape)
        ori_H,ori_W,_ = imgL.shape
        W=int(ori_W/2)
        H=int(ori_H/2)
        imgL=cv2.resize(imgL,(W,H),interpolation=cv2.INTER_LINEAR)
        imgR=cv2.resize(imgR,(W,H),interpolation=cv2.INTER_LINEAR)

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1 = 8 * img_channel * window_size ** 2,
            P2 = 32 * img_channel * window_size ** 2,
            disp12MaxDiff = 12,
            preFilterCap = 0,
            uniquenessRatio = 12,
            speckleWindowSize = 60, # 60
            speckleRange = 32, # 32
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
    elif method == 'BM':
        imgL=cv2.imread(imgL, cv2.IMREAD_GRAYSCALE)
        imgR=cv2.imread(imgR, cv2.IMREAD_GRAYSCALE)
        left_matcher = cv2.StereoBM_create(numDisparities=max_disp,
                                        blockSize=window_size)
    # time0 = time.time()
    displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16

    # dispr = left_matcher.compute(imgR_flip, imgL_flip).astype(np.float32)/16
    # dispr = np.flip(dispr,1)


    # print('runtime: ',time.time()-time0)
    return displ

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims,divide,mode=None):
        self.mode = mode
        self.ht, self.wd = dims[-2:]
        self.pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
        self.pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
        # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        # if mode == 'sintel':
        #     self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        # else:
        #     self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]
        self._pad = [0, self.pad_wd, 0, self.pad_ht]

    def pad(self, *inputs):
        # if self.mode is None:
        #     return [F.pad(x, self._pad) for x in inputs]
        # else:
        #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
        # return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs],self._pad[1],self._pad[3]

    def pad_numpy(self, *inputs):
        # if self.mode is None:
        #     return [F.pad(x, self._pad) for x in inputs]
        # else:
        #     return [F.pad(x, self._pad, mode=self.mode) for x in inputs]
        # return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        return [np.pad(x,((0,self.pad_ht),(0,self.pad_wd),(0,0))) for x in inputs],self._pad[1],self._pad[3]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        # c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def io_disp_read(dir):   
    '''
    function: load disparity map from disparity file
    input:
        dir: dir of disparity file
    ''' 
    # load disp from npy
    if dir.endswith('npy'):
        disp = np.load(dir)
        disp = disp.astype(np.float32)
    elif dir.endswith('pfm'):
        with open(dir, 'rb') as pfm_file:
            header = pfm_file.readline().decode().rstrip()
            channels = 3 if header == 'PF' else 1
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")
            scale = float(pfm_file.readline().decode().rstrip())
            if scale < 0:
                endian = '<' # littel endian
                scale = -scale
            else:
                endian = '>' # big endian
            disp = np.fromfile(pfm_file, endian + 'f')
            disp = np.reshape(disp, newshape=(height, width, channels))  
            disp[np.isinf(disp)] = 0
            disp = np.flipud(disp) 
            if channels == 1:
                disp = disp.squeeze(2)
        disp = disp.astype(np.float32)
    elif dir.endswith('png'):
        if 'Cre' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
            disp = disp.astype(np.float32) / 32
        elif 'middlebury' in dir:
            disp = cv2.imread(dir, -1)
            disp = disp.astype(np.float32)
        elif 'KITTI' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_ANYDEPTH) / 256.0
            disp = disp.astype(np.float32)
        elif 'real_road' in dir:
            _bgr = cv2.imread(dir)
            R_ = _bgr[:, :, 2]
            G_ = _bgr[:, :, 1]
            B_ = _bgr[:, :, 0]
            normalized_= (R_ + G_ * 256. + B_ * 256. * 256.) / (256. * 256. * 256. - 1)
            disp = 500*normalized_
            disp = disp.astype(np.float32)
        elif 'vkitti' in dir:
            # read depth
            depth = cv2.imread(dir, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # in cm
            depth = (depth / 100).astype(np.float32)  # depth clipped to 655.35m for sky
            valid = (depth > 0) & (depth < 655)  # depth clipped to 655.35m for sky
            # convert to disparity
            focal_length = 725.0087  # in pixels
            baseline = 0.532725  # meter
            disp = baseline * focal_length / depth
            disp[~valid] = 0  # invalid as very small value    
        elif 'Tongji' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_ANYDEPTH) / 256.0
            disp = disp.astype(np.float32)  
        if 'zhb' in dir:
            disp = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
            disp = disp.astype(np.float32) / 32  
    else:
        raise ValueError('unknown disp file type: {}'.format(dir))
    return disp

# recover the resolution of "input" from "reference"
def reso_recover(input,reference_dir):
    ori_img = cv2.imread(reference_dir)
    img_H,img_W = ori_img.shape[0],ori_img.shape[1]
    input = cv2.resize(input*img_W/input.shape[1],(img_W,img_H),interpolation=cv2.INTER_LINEAR)
    return input

def seed_record(img_W,img_H,Key_points_coordinate,disp_min=0):
    image1_seed = torch.zeros((img_H,img_W),device='cuda').long()
    # image2_seed = torch.zeros((img_H,img_W))
    co_row = Key_points_coordinate[:,1] == Key_points_coordinate[:,3]
    positive_disp = Key_points_coordinate[:,0] > Key_points_coordinate[:,2]
    saved = torch.logical_and(co_row,positive_disp)
    Key_points_coordinate = Key_points_coordinate[saved,:]
    image1_seed[Key_points_coordinate[:,1],Key_points_coordinate[:,0]]=Key_points_coordinate[:,0]-Key_points_coordinate[:,2]
    return image1_seed #,image2_seed

# resample the sim: coordinate based <--> disparity based
def sim_remap(sim):
    image_x = sim.shape[-1]
    w_base,d_base = torch.meshgrid(torch.from_numpy(np.arange(image_x)),torch.from_numpy(np.arange(image_x)))    
    d_base = w_base - d_base
    w_base = ((w_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    d_base = ((d_base)/(image_x-1)).unsqueeze(0).to(sim.device)
    coords = torch.stack((d_base, w_base), dim=3)
    # re-coordinate sim from [H,W,W] to [H,W,D]
    sim = F.grid_sample(sim.unsqueeze(0), 2*coords - 1, mode='nearest',
                            padding_mode='zeros',align_corners=True).squeeze()
    sim[sim==0] = -1
    return sim

# return points in [n,2] [width,height]
def SparseDisp2Points(disp,remove_margin=False):

    if remove_margin:
        # remove the correspondences at the image margin
        disp[:,0]=0
        disp[:,-1]=0
        disp[0,:]=0
        disp[-1,:]=0
        # grid = torch.arange(0, disp.shape[1], device='cuda').unsqueeze(0).expand(disp.shape[0],disp.shape[1]) # [H,W]: H * 0~W
        # disp[disp==grid]=0

    _ = disp.nonzero() # [n,2]

    disp = disp[disp>0]
    points_A = torch.zeros(_.shape).cuda()
    points_B = torch.zeros(_.shape).cuda()
    points_A[:,1] = _[:,0]
    points_B[:,1] = _[:,0]
    points_A[:,0] = _[:,1]
    points_B[:,0] = points_A[:,0]-disp

    return (points_A.long().t(),points_B.long().t())

# return disp
def Points2SparseDisp(H,W,points_A,points_B):
    # point [W,H]
    disp=torch.zeros(size=(H,W)).long().cuda()
    disp[points_A[1,:],points_A[0,:]]=points_A[0,:]-points_B[0,:]
    return disp

# confidence matrix initialization 
def sim_construct(feature_A,feature_B,LR = False):
        d1 = feature_A/torch.sqrt(torch.sum(torch.square(feature_A), 0)).unsqueeze(0) # [C,H,W]
        d2 = feature_B/torch.sqrt(torch.sum(torch.square(feature_B), 0)).unsqueeze(0) # [C,H,W]
        sim = torch.einsum('ijk,ijh->jkh', d1, d2) # [H,W,W] 166,240,240
        if LR:
            sim_l = sim_remap(sim) # convert to disparity
            sim_r = sim.permute(0,2,1).contiguous()
            sim_r = sim_r.flip([-1,-2])
            sim_r = sim_remap(sim_r) # convert to disparity
            return sim_l,sim_r
        else:
            return sim_remap(sim) 


def get_pt_disp(image_y,image_x,points=None,disp=None,offset=None,ratio=1):
    # disp should be numpy
    assert not isinstance(disp, torch.Tensor)
    assert points is None or disp is None
    if points is None:
        _ = disp.nonzero()
        u = _[1]
        v = _[0]
        dxs = disp[_]
    else:
        u = points[0] # column # width
        v = points[1] # row # height
        dxs = points[2] # disp
    PT_disp = getPT(u,v,dxs,image_y,image_x)
    PT_disp = PT_disp*ratio
    for j in range(image_y):
        ans = np.min(PT_disp[j,:])
        PT_disp[j,:] = ans
    if not offset is None:
        PT_disp = PT_disp - offset
    PT_disp[PT_disp<0]=0
    return PT_disp

def getPT(u,v,d,vmax,umax):
    v_map_1 = np.mat(np.arange(0, vmax)) # 
    v_map_1_transpose = v_map_1.T # (1030, 1)
    umax_one = np.mat(np.ones(umax)).astype(int) # (1, 1720)
    v_map = v_map_1_transpose * umax_one # (1030, 1720)
    vmax_one = np.mat(np.ones(vmax)).astype(int)
    vmax_one_transpose = vmax_one.T # (1030, 1)
    u_map_1 = np.mat(np.arange(0, umax)) # (1, 1720)
    u_map = vmax_one_transpose * u_map_1 # (1030, 1720)
    Su = np.sum(u)
    Sv = np.sum(v)
    Sd = np.sum(d)
    Su2 = np.sum(np.square(u))
    Sv2 = np.sum(np.square(v))
    Sdu = np.sum(np.multiply(u, d))
    Sdv = np.sum(np.multiply(v, d))
    Suv = np.sum(np.multiply(u, v))
    n= len(u)
    beta0 = (np.square(Sd) * (Sv2 + Su2) - 2 * Sd * (Sv * Sdv + Su * Sdu) + n * (np.square(Sdv) + np.square(Sdu)))/2
    beta1 = (np.square(Sd) * (Sv2-Su2) + 2 * Sd * (Su*Sdu-Sv*Sdv) + n * (np.square(Sdv) - np.square(Sdu)))/2
    beta2 = -np.square(Sd) * Suv + Sd * (Sv * Sdu + Su * Sdv) - n * Sdv * Sdu
    gamma0 = (n * Sv2 + n * Su2 - np.square(Sv) - np.square(Su))/2
    gamma1 = (n * Sv2 - n * Su2 - np.square(Sv) + np.square(Su))/2
    gamma2 = Sv * Su - n * Suv
    A = (beta1 * gamma0 - beta0 * gamma1)
    B = (beta0 * gamma2 - beta2 * gamma0)
    C = (beta1 * gamma2 - beta2 * gamma1)
    delta = np.square(A) + np.square(B) - np.square(C)
    tmp1 = (A + np.sqrt(delta))/(B-C)
    tmp2 = (A - np.sqrt(delta))/(B-C)
    theta1 = math.atan(tmp1)
    theta2 = math.atan(tmp2)
    u=np.mat(u)
    v=np.mat(v)
    d=np.mat(d)
    d=d.T
    u=u.T
    v=v.T
    t1 = v * math.cos(theta1) - u * math.sin(theta1)
    t2 = v * math.cos(theta2) - u * math.sin(theta2)
    n_ones = np.ones(n).astype(int)
    n_ones = (np.mat(n_ones)).T
    T1 = np.hstack((n_ones, t1))
    T2 = np.hstack((n_ones, t2))
    f1 = d.T * T1 * np.linalg.inv (T1.T * T1) * T1.T * d
    f2 = d.T * T2 * np.linalg.inv (T2.T * T2) * T2.T * d
    if f1 < f2:
        theta = theta2
    else:
        theta = theta1
    t = v * math.cos(theta) - u * math.sin(theta)
    T = np.hstack((n_ones, t))
    a = np.linalg.inv(T.T * T) * T.T * d
    t_map = v_map * math.cos(theta) - u_map * math.sin(theta)
    newdisp = (a[0] + np.multiply(a[1], t_map))# - 20
    return newdisp


def get_ncc_sim(left, right, ncc_rad = 4, max_disp = None):
    _,C,H,W = left.shape
    if max_disp is None:
        max_disp = W

    # ncc initial 
    ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

    # avg
    left_avg = ncc_pool(left_padded) # [1,C,H,W]
    right_avg = ncc_pool(right_padded)
    left_avg = left_avg.unsqueeze(2) # [1,C,1,H,W]
    right_avg = right_avg.unsqueeze(2) 
    # unfold
    left_sum = ncc_Unfold(left_padded)
    left_sum = left_sum.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    right_sum = ncc_Unfold(right_padded)
    right_sum = right_sum.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # minus
    left_minus = left_sum-left_avg # [1,C,rad^,H,W]
    right_minus = right_sum-right_avg

    var_left = torch.sum(torch.square(left_minus),dim=2) # [1,C,H,W]
    var_left[var_left==0]=0.01
    var_right = torch.sum(torch.square(right_minus),dim=2) # [1,C,H,W]
    var_right[var_right==0]=0.01
    
    # calculate
    ncc_b = torch.matmul(var_left.unsqueeze(-1),var_right.unsqueeze(-2)) # [1,C,H,W,W]
    ncc_b = torch.sqrt(ncc_b) # [1,C,H,W,W]

    # ncc_a =  torch.matmul(left_minus.permute(0,1,3,4,2).contiguous(),right_minus.permute(0,1,3,2,4).contiguous()) # [1,C,H,W,W] 
    Conf = (torch.matmul(left_minus.permute(0,1,3,4,2).contiguous(),right_minus.permute(0,1,3,2,4).contiguous()))/ncc_b # [1,C,H,W,W] 
    Conf = torch.mean(Conf,dim=1) # [1,H,W,W]

    Conf = sim_remap(Conf.squeeze()) # [H,W,W]
    Conf = Conf[:,:,:max_disp] 

    return Conf

### ranger ###
def get_ncc_sim(left, right, ncc_rad = 3, max_disp = None, max_disp_ratio = 0.25):
    
    B,C,H,W = left.shape
    N = (2*ncc_rad+1)*(2*ncc_rad+1)
    if max_disp is None:
        max_disp = int(W*max_disp_ratio)
    Conf = left.new_zeros(H,W,max_disp)-1

    # ncc initial 
    ncc_pool =  torch.nn.AvgPool2d((ncc_rad*2+1), stride=1,padding=0)
    ncc_Unfold=torch.nn.Unfold(kernel_size=((ncc_rad*2+1),(ncc_rad*2+1)),dilation=1,padding=0,stride=1)
    # pad
    left_padded = F.pad(left, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]
    right_padded = F.pad(right, [ncc_rad,ncc_rad,ncc_rad,ncc_rad],mode='replicate') # [1,1, H，W]

    # avg
    left_avg = ncc_pool(left_padded) # [1,C,H,W]
    right_avg = ncc_pool(right_padded)
    # unfold
    left_unfold = ncc_Unfold(left_padded)
    left_unfold = left_unfold.view(B,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # left_unfold = left_unfold.permute(0,1,3,4,2).contiguous()
    right_unfold = ncc_Unfold(right_padded)
    right_unfold = right_unfold.view(1,C,(ncc_rad*2+1)*(ncc_rad*2+1),H,W) # [1,C,rad^,H,W]
    # right_unfold = right_unfold.permute(0,1,3,2,4).contiguous()
    # var
    left_minus = left_unfold-left_avg.unsqueeze(2) # [1,C,rad^,H,W]
    right_minus = right_unfold-right_avg.unsqueeze(2)
    var_left = torch.sum(torch.square(left_minus),dim=2) # [1,C,H,W]
    # var_left[var_left==0]=0.01
    var_left = var_left+1e-6
    var_right = torch.sum(torch.square(right_minus),dim=2) # [1,C,H,W]
    # var_right[var_right==0]=0.01
    var_right = var_right+1e-6

    conf_ = ((left_unfold[:,:,:,:,:]*right_unfold[:,:,:,:,:]).sum(dim=2) - N*left_avg[:,:,:,:]*right_avg[:,:,:,:])/torch.sqrt(var_left[:,:,:,:]*var_right[:,:,:,:])
    Conf[:, 0:, 0] = torch.mean(conf_,dim=1).squeeze()
    for i in range(1,max_disp):
        conf_ = ((left_unfold[:,:,:,:,i:]*right_unfold[:,:,:,:,:-i]).sum(dim=2) - N*left_avg[:,:,:,i:]*right_avg[:,:,:,:-i])/torch.sqrt(var_left[:,:,:,i:]*var_right[:,:,:,:-i])
        Conf[:, i:, i] = torch.mean(conf_,dim=1).squeeze()
    return Conf

def writePFM(file, image, scale=1):
    file = open(file, 'wb')
 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    image = np.flipud(image)
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
 
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n'.encode() % scale)
 
    image.tofile(file)

def save_disp_results(disp_dir,png_dir,result,max_disp=None,min_disp=None,display=True,PLA=False):
    if display:
        print('save in {} and {}, shape = {}'.format(png_dir,disp_dir,result.shape))
    if disp_dir is not None:
        if 'npy' in disp_dir:
            np.save(disp_dir, result)
        elif 'pfm' in disp_dir:
            writePFM(disp_dir,result.astype(np.float32))
    if png_dir is not None:
        if not PLA:
            disp_vis(png_dir,result,max_disp,min_disp)
        else:
            disp_vis(png_dir,result,max_disp,min_disp,colormap=cv2.COLORMAP_PLASMA)

def sim_down_size(sim,down_size=1):
    img_x = sim.shape[-1]
    max_disp = int(img_x/down_size)
    # max_disp = img_x
    sim = sim[:,:,:max_disp]
    return sim

def sim_restore(sim,value=-1):
    H,W,D = sim.shape
    if W==D:
        return sim
    expand = torch.zeros((H,W,W-D),device='cuda')-1
    expand = expand+value
    sim = torch.cat((sim,expand),-1)
    return sim
    
def KITTI_test_submission(save_dir,disp):
    save_dir = save_dir.replace('vis','submitted/disp_0')
    save_dir_base = '/'.join(save_dir.split('/')[:-1])
    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)
    disp = np.clip(disp,0,255)
    disp = disp*256.
    disp = disp.astype(np.uint16)
    skimage.io.imsave(save_dir, disp)

def middeval3_submission(save_dir,disp):
    
    # print(save_dir.shape)
    # exit()
    
    split = save_dir.split('/')
    method = split[-2]
    save_dir = '/'.join(split[:4])+'/submit'+'/'+split[4]+'/{}'.format(method)+'/'+split[5]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,'disp0{}.pfm'.format(method))
    save_disp_results(save_path,None,disp,display=False)
    save_path_time = os.path.join(save_dir,'time{}.txt'.format(method))
    with open(save_path_time,'w') as f:
        f.write('66.660000')

def test_submission(save_dir,disp):
    if 'KITTI' in save_dir and 'testing' in save_dir:
        KITTI_test_submission(save_dir,disp)
    # if 'MiddEval3' in save_dir and 'test' in save_dir:
    #     middeval3_submission(save_dir,disp)
    if 'MiddEval3' in save_dir:
        middeval3_submission(save_dir,disp)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def transparent_generate(bg=None,fg=None,bg_dir=None,fg_dir=None,save_dir='generate_iamges/transparent_image.png'):
    if bg is None:
        bg_img = cv2.imread(bg_dir)
    if fg is None:
        fg_img = cv2.imread(fg_dir)
    assert fg_img.shape==bg_img.shape
    result = cv2.addWeighted(bg_img,0.3,fg_img,0.7,gamma=0)
    cv2.imwrite(save_dir,result)


def SLIC_opencv(img,save_dir=None,region_size=10,ruler=10.0,iteration=10,HSV=False,visualize=False):
    # img = cv2.imread(batch['left_dir'][0], flags=1)  # 读取彩色图像(BGR)
    if HSV:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)  # BGR-HSV 转换
    # SLIC 算法
    # slic = cv2.ximgproc.SuperpixelSLIC(img, region_size=10, ruler=10.0)  # 初始化 SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, ruler=ruler)  # 初始化 SLIC
    slic.iterate(iteration)  # 迭代次数，越大效果越好
    label_slic = slic.getLabels()  # 获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目

    if visualize:
        assert not save_dir is None
        
        mask_slic = slic.getLabelContourMask()  # 获取 Mask，超像素边缘 Mask==1
        mask_color = np.array([mask_slic for i in range(3)]).transpose(1, 2, 0)  # 转为 3 通道
        # mask_color= cv2.COLOR_GRAY2RGB(mask_slic)  # 灰度 Mask 转为 RGB
        img_slic = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_slic))  # 在原图上绘制超像素边界
        # print(img_slic.shape)
        imgSlic = cv2.add(img_slic, mask_color)
        
        plt.figure(figsize=(9, 7))
        plt.subplot(221), plt.axis('off'), plt.title("Origin image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 显示 img(RGB)
        plt.subplot(222), plt.axis('off'), plt.title("SLIC mask")
        plt.imshow(mask_slic, 'gray')
        plt.subplot(223), plt.axis('off'), plt.title("SLIC image")
        plt.imshow(cv2.cvtColor(img_slic, cv2.COLOR_BGR2RGB))
        plt.subplot(224), plt.axis('off'), plt.title("SLIC image")
        plt.imshow(cv2.cvtColor(imgSlic, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_dir,dpi=800)
        plt.close()

    return label_slic,number_slic

def default_leastsq_error(p, x, y):
    return np.square(p[0]*x+p[1] - y)    

def least_sq(array,target_array,valid=None,error=default_leastsq_error):
    if valid is None:
        array = array.flatten()
        target_array = target_array.flatten()
    else:
        array = array[valid]
        target_array = target_array[valid]
    
    Para = leastsq(error, np.array([1, 0]), args=(array, target_array))
    k, b = Para[0]
    return(k,b)

# convert an numpy array to cv2 int8 img
def norm_255(img):
    max = np.max(img)
    min = np.min(img)
    img = (img - min) / (max - min)
    img = img * 255
    return img.astype('uint8')


def get_FBS(img,cost_volume,rad=1,gamma_d=1,gamma_r=7,cycle=1):

    # print('rad',rad)
    # print('cycle',cycle)

    if cycle > 0:
        # gamma_d = 15
        gamma_r = torch.tensor(gamma_r) # BGNet:7; Unimatch:7; LacGwc:7; AANet:4; PSMNet:7; Cre:9

        assert rad in [1,2,3,4,5,6]
        img_y,img_x,D =cost_volume.shape # H,W,D
        pad = rad

        cost_volume = cost_volume.permute(2,0,1).unsqueeze(0) # [1,D,H,W]
        mask = cost_volume==-1

        img = F.interpolate(img, size=(img_y,img_x), mode='bilinear', align_corners=True)
        img_pad = F.pad(img, [pad,pad,pad,pad],mode='replicate') # [1,1, H，W]
        unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=1,padding=0,stride=1)
        img_unfold = unfold(img_pad)
        img_unfold = img_unfold.view((2*rad+1)**2,img_y,img_x).permute(1,2,0) #[H,W,9]
        # gamma_r = torch.std(img_unfold,dim=-1,keepdim=True)*2
        w_r = torch.exp(-torch.square(img_unfold - img.squeeze().unsqueeze(-1))/(2*torch.square(gamma_r))) # [H,W,9]
        # if rad == 1:
        #     w_d = torch.tensor([[2,1,2],[1,0,1],[2,1,2]]).cuda()
        # elif rad == 2:
        #     w_d = torch.tensor([[8,5,4,5,8],[5,2,1,2,5],[4,1,0,1,4],[5,2,1,2,5],[8,5,4,5,8]]).cuda()
        # w_d = torch.exp(-w_d/gamma_d**2).view(-1).unsqueeze(0).unsqueeze(0) # [1,1,9]
        # w = (w_r*w_d).unsqueeze(-2) # [H,W,1,9]
        w = w_r.permute(-1,0,1).unsqueeze(0) # [1,9,H,W]
        w /= torch.sum(w,dim=1).unsqueeze(0) # [1,9,H,W]

        for i in range(cycle):
            cost_volume = F.pad(cost_volume,[pad,pad,pad,pad],mode='replicate') # [1,D,H+rad,W+rad]
            cost_volume = unfold(cost_volume)
            cost_volume = cost_volume.view(D,(2*rad+1)**2,img_y,img_x) #[D,9,H,W]
            cost_volume = cost_volume*w
            cost_volume = torch.sum(cost_volume,dim=1).unsqueeze(0) # [1,D,H,W]
            cost_volume[mask]=-1
        
        cost_volume = cost_volume.squeeze(0).permute(1,2,0)

    return cost_volume

# larger sigma, all weights seem the same
def distance_weight_init(rad,B,C,H,W,device='cuda'):
    x_base = torch.linspace(0, 2*rad, 2*rad+1, device=device).unsqueeze(0).repeat(2*rad+1, 1)
    y_base = torch.linspace(0, 2*rad, 2*rad+1, device=device).unsqueeze(1).repeat(1,2*rad+1)
    weight = torch.square(x_base-rad)+torch.square(y_base-rad) # [2*rad+1,2*rad+1]
    weight = weight.view(-1) # N
    weight = weight.repeat(B,C,H,W,1)
    return weight

def calcu_D1(pre,gt):
    abs_diff = torch.abs(gt - pre)
    error = torch.ones(abs_diff.shape)
    total = torch.sum(error)
    error[abs_diff < gt*0.05] = 0
    error[abs_diff < 3] = 0
    num_error = torch.sum(error)
    ans = num_error/total
    return ans
    
    
def calcu_PEP(pre,gt,thr):
    abs_diff = torch.abs(gt - pre)
    error = torch.ones(abs_diff.shape)
    total = torch.sum(error)
    error[abs_diff < thr] = 0
    num_error = torch.sum(error)
    ans = num_error/total
    return ans
    