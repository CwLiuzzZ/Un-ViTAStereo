import torch
import cv2
import torch.nn.functional as F

def gradient_single(input):
    dim = len(input.shape)
    assert dim in [3,4]
    input = F.pad(input, [1,1,1,1], mode='replicate')
    grad_x = torch.abs(input - torch.roll(input,1,0))
    grad_y = torch.abs(input - torch.roll(input,1,1))
    if dim == 3:
        grad_x = grad_x[:,1:-1,1:-1]
        grad_y = grad_y[:,1:-1,1:-1]
    elif dim == 4:
        grad_x = grad_x[:,:,1:-1,1:-1]
        grad_y = grad_y[:,:,1:-1,1:-1]
    return grad_x,grad_y

def gradient_mutual(input,return_matrix=False,abs=True):  
    dim = len(input.shape)
    assert dim in [3,4]
    input = F.pad(input, [1,1,1,1], mode='replicate')
    v_g_1 = input - torch.roll(input,1,-2)
    v_g_2 = input - torch.roll(input,-1,-2)
    u_g_1 = input - torch.roll(input,1,-1)
    u_g_2 = input - torch.roll(input,-1,-1)
    g_matrix = torch.stack([v_g_1,v_g_2,u_g_1,u_g_2],-1) # [...,4]
    if abs:
        g_matrix = torch.abs(g_matrix)
    if dim == 3:
        g_matrix = g_matrix[:,1:-1,1:-1,:]
    elif dim == 4:
        g_matrix = g_matrix[:,:,1:-1,1:-1,:]
    if return_matrix:
        return g_matrix # [...,4]
    g_max_matrix = torch.max(g_matrix,dim=-1)[0]
    g_min_matrix = torch.min(g_matrix,dim=-1)[0]
    # g_max = g_max_matrix.mean()
    return g_max_matrix,g_min_matrix 

def gradient_window(input,abs=True,rad=1):  
    assert len(input.shape) == 4
    B,C,H,W = input.shape
    N = (2*rad+1)**2
    padded = F.pad(input, [rad,rad,rad,rad], mode='replicate')
    nn_Unfold=torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=1,padding=0,stride=1)
    unfold = nn_Unfold(padded)
    unfold = unfold.view(B,C,N,H,W).permute(0,1,3,4,2) #[B,C,H,W,N]
    grad = unfold-input.unsqueeze(-1)
    if abs:
        return torch.abs(grad)
    else:
        return grad


# todo for batch operation
def depth_outlier_filter(depth): 
    depth = depth + 1 
    h = depth.shape[0]  
    w = depth.shape[1] 
    max_ = torch.max(depth)+1
    v_g_1 = torch.abs(depth - torch.roll(depth,1,0))
    v_g_2 = torch.abs(depth - torch.roll(depth,-1,0))
    u_g_1 = torch.abs(depth - torch.roll(depth,1,1))
    u_g_2 = torch.abs(depth - torch.roll(depth,-1,1))
    g_matrix = torch.stack([v_g_1,v_g_2,u_g_1,u_g_2],-1)
    
    g_max_matrix = torch.max(g_matrix,dim=-1)[0]
    g_min_matrix = torch.min(g_matrix,dim=-1)[0]
    g_max = g_max_matrix.mean()
    error_mask = g_min_matrix>g_max # True for error pixels # indenty outliers
    cv2.imwrite('generate_images/delete_max_0.png',error_mask.detach().cpu().numpy()*255)
        
    error_mask = (~error_mask).float() # 0 for error pixels
    rad = 10 # radius
    unfold = torch.nn.Unfold(kernel_size=(rad*2+1,rad*2+1),dilation=1,padding=0,stride=1)
    depth_unfold = F.pad(depth.unsqueeze(0), [rad,rad,rad,rad],mode='constant',value=max_) 
    error_mask_unfold = F.pad(error_mask.unsqueeze(0), [rad,rad,rad,rad],mode='constant',value=0.0) 
    depth_unfold = unfold(depth_unfold).view((rad*2+1)*(rad*2+1),h,w).permute(1,2,0) #[H,W,window_size]
    error_mask_unfold = unfold(error_mask_unfold).view((rad*2+1)*(rad*2+1),h,w).permute(1,2,0) #[H,W,window_size] # 0 for error pixels
    assert torch.min(torch.max(error_mask_unfold,-1)[0])==1,"exist window of all error pixels, please enlarge the unfold window"
    depth_unfold[error_mask_unfold==0]=max_
    depth_diff = torch.abs(depth_unfold-depth.unsqueeze(-1)) # [H,W,window_size]
    _, min_index = torch.min(depth_diff, dim=-1) 
    update_depth = torch.gather(depth_unfold,-1,min_index.unsqueeze(-1)).squeeze() 
    return update_depth


# input be [B,C,H,W]
def tensor_warp(source,disp,mode='right'):
    assert mode in ['left','right']
    assert isinstance(source, torch.Tensor) and isinstance(disp, torch.Tensor)
    assert len(source.shape) == 4 and len(disp.shape) == 4
    assert source.shape[-1] == disp.shape[-1]
    assert source.shape[-2] == disp.shape[-2]
    assert disp.shape[1] == 1
    B,_,H,W = source.shape
    
    disp = (disp/disp.shape[-1]).float()

    if mode == 'left':
        if torch.mean(disp)<0:
            disp=-disp
    elif mode == 'right':
        if torch.mean(disp)>0:
            disp=-disp
            
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, W).repeat(B,H, 1).type_as(source)
    y_base = torch.linspace(0, 1, H).repeat(B,W, 1).transpose(1, 2).type_as(source)
    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with C=1
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(source, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros',align_corners=True)

    return output # [B,C,H,W]
