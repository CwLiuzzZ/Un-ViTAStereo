import torch
from torch import nn
from toolkit.function.base_function import distance_weight_init


class Local_Ranking_Loss(nn.Module): # LoRa Version3
    def __init__(self, rad=5, dilation=6,topK=8):
        super(Local_Ranking_Loss, self).__init__()
        self.rad = rad
        self.dilation = dilation
        self.N = (2*rad+1)**2
        self.padding = rad*dilation
        self.nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=dilation,padding=0,stride=1)
        # self.BCE = nn.BCELoss()
        self.Softshrink = nn.Softshrink(1)
        self.Softsign = nn.Softsign()
        self.Relu = nn.ReLU()
        self.topK = topK

    # lrc_mask: False for invalid disparities
    def forward(self,disp,depth,vote):
        B,C,H,W = disp.shape
        
        # generate paired depth and disp
        depth_padded = nn.functional.pad(depth, [self.padding,self.padding,self.padding,self.padding], mode='replicate')
        disp_padded = nn.functional.pad(disp, [self.padding,self.padding,self.padding,self.padding], mode='replicate')
        vote = nn.functional.pad(vote, [self.padding,self.padding,self.padding,self.padding], mode='constant', value=1.0)
        
        vote = self.nn_Unfold(vote)
        vote = vote.view(B,C,self.N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
        vote_score,index = torch.topk(vote, k=self.topK, largest=False) # [B,C,H,W,K]
        
        depth_Gradient = self.nn_Unfold(depth_padded)
        depth_Gradient = depth_Gradient.view(B,C,self.N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
        depth_Gradient = depth.unsqueeze(-1) - depth_Gradient
        depth_Gradient = torch.gather(depth_Gradient,-1,index) # [B,C,H,W,K]
        disp_Gradient = self.nn_Unfold(disp_padded)
        disp_Gradient = disp_Gradient.view(B,C,self.N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
        disp_Gradient = disp.unsqueeze(-1) - disp_Gradient # [B,C,H,W,N]
        disp_Gradient = torch.gather(disp_Gradient,-1,index) # [B,C,H,W,K]
        
        # direction = (disp_direction/torch.abs(disp_direction))*(depth_direction/torch.abs(depth_direction)) # 1 for same direction, -1 for error
        disp_direct = disp_Gradient/(torch.abs(disp_Gradient)+1e-8) # [B,C,H,W,K]
        
        depth_weight = self.Softshrink(depth_Gradient) # todo if use softshrink # [B,C,H,W,K]
        # print('depth_weight',torch.mean(depth_weight))
        depth_weight = self.Softsign(depth_weight) # [B,C,H,W,K]
        
        disp_value = torch.log(1+torch.square(disp_Gradient)) # [B,C,H,W,K]
        
        loss = self.Relu(-depth_weight*disp_direct*disp_value).mean() 
        
        return loss

    
def LoRa_check(disp,depth,vote,rad=5, dilation=10, topK=8):
    N = (2*rad+1)**2
    padding = rad*dilation
    nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=dilation,padding=0,stride=1)
    Softshrink = nn.Softshrink(1)
    Relu = nn.ReLU()

    B,C,H,W = disp.shape
    
    # generate paired depth and disp
    depth_padded = nn.functional.pad(depth, [padding,padding,padding,padding], mode='replicate')
    disp_padded = nn.functional.pad(disp, [padding,padding,padding,padding], mode='replicate')
    vote = nn.functional.pad(vote, [padding,padding,padding,padding], mode='constant', value=1.0)
    
    vote = nn_Unfold(vote)
    vote = vote.view(B,C,N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
    _,index = torch.topk(vote, k=topK, largest=False)
    
    depth_Gradient = nn_Unfold(depth_padded)
    depth_Gradient = depth_Gradient.view(B,C,N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
    depth_Gradient = depth.unsqueeze(-1) - depth_Gradient
    depth_Gradient = torch.gather(depth_Gradient,-1,index) # [B,C,H,W,topK]
    disp_Gradient = nn_Unfold(disp_padded)
    disp_Gradient = disp_Gradient.view(B,C,N,H,W).permute(0,1,3,4,2).contiguous() # [B,C,H,W,N]
    disp_Gradient = disp.unsqueeze(-1) - disp_Gradient # [B,C,H,W,N]
    disp_Gradient = torch.gather(disp_Gradient,-1,index) # [B,C,H,W,topK]
    
    # direction = (disp_direction/torch.abs(disp_direction))*(depth_direction/torch.abs(depth_direction)) # 1 for same direction, -1 for error
    disp_direct = disp_Gradient/(torch.abs(disp_Gradient)+1e-8) # [B,C,H,W,topK]
    # disp_direct = self.Softsign(disp_direct)
    
    depth_weight = Softshrink(depth_Gradient) # todo if use softshrink # [B,C,H,W,topK]
    # print('depth_weight',torch.mean(depth_weight))
    # depth_weight = self.Softsign(depth_weight)
    # depth_weight = depth_weight*torch.abs(depth_weight)
    
    disp_value = torch.log(1+torch.square(disp_Gradient)) # [B,C,H,W,topK]
    # disp_value = torch.log(1+torch.abs(disp_direction))
    # disp_value = torch.exp(torch.abs(disp_direction))
    
    loss = Relu(-depth_weight*disp_direct*disp_value).sum(dim=-1) # [B,C,H,W]
    
    return loss

# use only the pixel pairs with max and min depth gradient
def LoRa_MaxMin_Vote(disp,depth,rad=10, dilation=3, thr1=7, thr2=2, ratio=0.15):
    N = (2*rad+1)**2
    padding = rad*dilation
    nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=dilation,padding=0,stride=1)
    B,C,H,W = disp.shape
    # invalid the margin pixels
    valid_check = torch.ones(depth.shape,device=disp.device).bool() # pixel with valid ckeck
    valid_input = torch.ones(depth.shape,device=disp.device).float() # input disp with valid value
    # valid_output = torch.ones(depth.shape,device=disp.device).bool() # output valid disp after check23
    
    valid_check[disp==0] = False
    valid_input[disp==0] = 0
    
    # pad
    depth_padded = nn.functional.pad(depth, [padding,padding,padding,padding], mode='replicate')
    disp_padded  = nn.functional.pad(disp, [padding,padding,padding,padding], mode='replicate')
    valid_padded = nn.functional.pad(valid_input, [padding,padding,padding,padding], mode='constant',value=0.0)
    
    # unfold
    depth_unfold = nn_Unfold(depth_padded)
    depth_unfold = depth_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    disp_unfold  = nn_Unfold(disp_padded)
    disp_unfold  = disp_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    valid_unfold = nn_Unfold(valid_padded)
    valid_unfold = valid_unfold.view(B,C,N,H,W).permute(0,1,3,4,2).bool() # [B,C,H,W,N]
    
    # generate gradient
    depth_gradient = depth_unfold - depth.unsqueeze(-1) # [B,C,H,W,N]
    disp_gradient  = disp_unfold  -  disp.unsqueeze(-1) # [B,C,H,W,N]
    depth_gradient[~valid_unfold] = 0 # invalid the invalid input disparity in depth_gradient
    
    max_depth_gradient,max_depth_gradient_index = torch.max(depth_gradient,-1) # [B,C,H,W] # positive value: neighbor pixel has large depth
    min_depth_gradient,min_depth_gradient_index = torch.min(depth_gradient,-1) # [B,C,H,W] # negative value: neighbor pixel has small depth
    
    valid_check[max_depth_gradient <  thr2] = False
    valid_check[min_depth_gradient > -thr2] = False
    
    max_disp_gradient = torch.gather(disp_gradient,-1,max_depth_gradient_index.unsqueeze(-1)).squeeze(-1) # [B,C,H,W]
    min_disp_gradient = torch.gather(disp_gradient,-1,min_depth_gradient_index.unsqueeze(-1)).squeeze(-1) # [B,C,H,W]
    
    ratio_diff = torch.abs((-min_depth_gradient)/(max_depth_gradient-min_depth_gradient+1e-3) - (-min_disp_gradient)/(max_disp_gradient-min_disp_gradient+1e-3)) # [B,C,H,W]
    
    valid_mask = (ratio_diff < ratio) * (max_disp_gradient > 0) * (min_disp_gradient < 0) * valid_check
    disp[~valid_mask] = 0
    
    return disp,valid_mask

# use only the pixel pairs with max and min depth gradient
# larger sigma, all weights seem the same
def LoRa_Window_Vote_test(disp,depth,rad=6, thr = 5,dilation=4,Vote_ratio=0.2,sigma = 10, return_score=False):
    
    N = (2*rad+1)**2
    padding = rad*dilation
    nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=dilation,padding=0,stride=1)
    B,C,H,W = disp.shape
    valid_input = (disp>0).float() # input disp with valid value
    
    # unfold 
    depth_padded = nn.functional.pad(depth, [padding,padding,padding,padding], mode='replicate')
    disp_padded  = nn.functional.pad(disp, [padding,padding,padding,padding], mode='replicate')
    valid_padded = nn.functional.pad(valid_input, [padding,padding,padding,padding], mode='constant',value=0.0)
    depth_unfold = nn_Unfold(depth_padded)
    depth_unfold = depth_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    disp_unfold  = nn_Unfold(disp_padded)
    disp_unfold  = disp_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    valid_unfold = nn_Unfold(valid_padded)
    valid_unfold = valid_unfold.view(B,C,N,H,W).permute(0,1,3,4,2).bool() # [B,C,H,W,N] True for valid disp
    # support_count = torch.sum(valid_unfold,dim=-1).float() # [B,C,H,W] count the support (valid) pixels
    
    # generate gradient
    depth_gradient = depth_unfold - depth.unsqueeze(-1) # [B,C,H,W,N]
    disp_gradient  = disp_unfold  -  disp.unsqueeze(-1) # [B,C,H,W,N]
     
    ############################################
    ######### vote: True for against ###########
    ############################################
    # pixel pairs with enough depth variation but opposite disp gradient direction
    direct_check = torch.zeros(depth_gradient.shape,device=depth_gradient.device).float() # [B,C,H,W,N]
    direct_vote = torch.zeros(depth_gradient.shape,device=depth_gradient.device).bool() 
    direct_check[depth_gradient > 0.67*thr] = 1 # [B,C,H,W,N]
    direct_check[depth_gradient < -0.67*thr] = -1 # [B,C,H,W,N]
    direct_vote[direct_check * disp_gradient<0] = True # [B,C,H,W,N]
    # pixel pairs with large depth variation and small disp variation
    large_varia_vote = torch.logical_and(torch.abs(depth_gradient) > thr*2, torch.abs(disp_gradient) < 1) # [B,C,H,W,N]
    # pixel pairs with small depth variation and large disp variation
    small_varia_vote = torch.logical_and(torch.abs(depth_gradient) < thr, torch.abs(disp_gradient) > 2) # [B,C,H,W,N]
    # disproportionate depth and disparity variations:
    # ratio = torch.abs(depth_gradient)/(torch.abs(disp_gradient)+1e-8) # [B,C,H,W,N]
    # disproportionate =  torch.logical_and(torch.logical_and(torch.abs(depth_gradient) < 5*thr,torch.abs(depth_gradient) > 3*thr), torch.logical_or(ratio>thr*5,ratio<thr*0.2)) # [B,C,H,W,N]
    # ratio_vote = torch.abs(torch.abs(depth_gradient)/(torch.abs(disp_gradient)+1e-8) - thr)
     
    
    # put together the against vote
    against_vote = torch.logical_or(direct_vote,small_varia_vote) # [B,C,H,W,N]
    against_vote = torch.logical_or(against_vote,large_varia_vote) # [B,C,H,W,N]
    # against_vote = torch.logical_or(against_vote,disproportionate) # [B,C,H,W,N]
    against_vote[~valid_unfold] = False # only valid disp can vote
    
    # if not lrc_diff is None:
    #     lrc_diff_padded = nn.functional.pad(lrc_diff, [padding,padding,padding,padding], mode='replicate') # [B,C,H,W]
    #     lrc_diff_unfold = nn_Unfold(lrc_diff_padded)
    #     lrc_diff_unfold = lrc_diff_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    #     # lrc_diff_weight = torch.exp(-lrc_diff_unfold/(torch.square(torch.mean(lrc_diff)))) # [B,C,H,W,N]
    #     lrc_diff_weight = torch.exp(-lrc_diff_unfold/(8)) # [B,C,H,W,N]
    #     lrc_diff_weight[~valid_unfold] = 0  # invalid cannot vote
    #     lrc_diff_weight = lrc_diff_weight/torch.sum(lrc_diff_weight,dim=-1,keepdim = True)
    #     vote_score = against_vote*lrc_diff_weight
    #     vote_score = torch.sum(vote_score.float(),dim=-1) # [B,C,H,W]
    # else:
    # distance weight
    weight = distance_weight_init(rad,B,C,H,W) # [B,C,H,W,N]
    weight = weight/(sigma*sigma)
    weight =  torch.exp(-weight) # larger sigma, all weights seem the same
    weight[~valid_unfold] = 0 # invalid cannot vote
    weight = weight/torch.sum(weight,dim=-1,keepdim = True) # [B,C,H,W,N]
    vote_score = against_vote*weight # [B,C,H,W,N]
    vote_score = torch.sum(vote_score.float(),dim=-1) # [B,C,H,W]
    
    # # bi-derction depth check
    # biderc_check = depth_gradient.clone()
    # biderc_check[~valid_unfold] = 0
    # biderc_1 = (biderc_check>0).any(dim=-1)
    # biderc_2 = (biderc_check<0).any(dim=-1)
    # biderc_check = biderc_1 * biderc_2 # biderc_check == 0 for single direction 
    # vote_score[~biderc_check] = 1
    
    if return_score:
        return vote_score
    
    valid_mask = vote_score < Vote_ratio
    disp[~valid_mask] = 0
    
    return disp,valid_mask,vote_score

# use only the pixel pairs with max and min depth gradient
# larger sigma, all weights seem the same
def LoRa_Window_Vote_test_v2(disp,depth,thr,rad=6,dilation=4,sigma = 10):
    
    N = (2*rad+1)**2
    padding = rad*dilation
    nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=dilation,padding=0,stride=1)
    B,C,H,W = disp.shape
    valid_input = (disp>0).float() # input disp with valid value
    
    # unfold 
    depth_padded = nn.functional.pad(depth, [padding,padding,padding,padding], mode='replicate')
    disp_padded  = nn.functional.pad(disp, [padding,padding,padding,padding], mode='replicate')
    valid_padded = nn.functional.pad(valid_input, [padding,padding,padding,padding], mode='constant',value=0.0)
    depth_unfold = nn_Unfold(depth_padded)
    depth_unfold = depth_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    disp_unfold  = nn_Unfold(disp_padded)
    disp_unfold  = disp_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    valid_unfold = nn_Unfold(valid_padded)
    valid_unfold = valid_unfold.view(B,C,N,H,W).permute(0,1,3,4,2).bool() # [B,C,H,W,N] True for valid disp
    
    # thr_unfold = torch.mean(torch.abs(depth_unfold - depth.unsqueeze(-1)),-1)/torch.mean(torch.abs(disp_unfold - disp.unsqueeze(-1)),-1)
    # print(torch.mean(thr_unfold))
    # print(torch.max(thr_unfold))
    # print(torch.min(thr_unfold))
      
    # generate gradient
    depth_gradient = depth_unfold - depth.unsqueeze(-1) # [B,C,H,W,N]
    disp_gradient  = disp_unfold  -  disp.unsqueeze(-1) # [B,C,H,W,N]
    
    ############################################
    ######### vote: True for against ###########
    ############################################
    # pixel pairs with enough depth variation but opposite disp gradient direction
    direct_check = torch.zeros(depth_gradient.shape,device=depth_gradient.device).float() # [B,C,H,W,N]
    direct_vote = torch.zeros(depth_gradient.shape,device=depth_gradient.device).bool() 
    # direct_check[depth_gradient > 0.67*thr] = 1 # [B,C,H,W,N]
    # direct_check[depth_gradient < -0.67*thr] = -1 # [B,C,H,W,N]
    # direct_vote[(direct_check * disp_gradient)<0] = True # [B,C,H,W,N]
    direct_vote[(depth_gradient * disp_gradient)<0] = True # [B,C,H,W,N]
    # pixel pairs with large depth variation and small disp variation
    large_varia_vote = torch.logical_and(torch.abs(depth_gradient) > thr*2, torch.abs(disp_gradient) < 1) # [B,C,H,W,N]
    # pixel pairs with small depth variation and large disp variation
    small_varia_vote = torch.logical_and(torch.abs(depth_gradient) < thr, torch.abs(disp_gradient) > 2) # [B,C,H,W,N]
        
    
    # put together the against vote
    against_vote = torch.logical_or(direct_vote,small_varia_vote) # [B,C,H,W,N]
    against_vote = torch.logical_or(against_vote,large_varia_vote) # [B,C,H,W,N]
    # against_vote = torch.logical_or(against_vote,disproportionate) # [B,C,H,W,N]
    against_vote[~valid_unfold] = False # only valid disp can vote
    
    weight = distance_weight_init(rad,B,C,H,W) # [B,C,H,W,N]
    weight = weight/(sigma*sigma)
    weight =  torch.exp(-weight) # larger sigma, all weights seem the same
    weight[~valid_unfold] = 0 # invalid cannot vote
    weight = weight/torch.sum(weight,dim=-1,keepdim = True) # [B,C,H,W,N]
    vote_score = against_vote*weight # [B,C,H,W,N]
    vote_score = torch.sum(vote_score.float(),dim=-1) # [B,C,H,W]
        
    return vote_score

# def margin_diffusion_check(disp,depth,thr1=7,thr2=2,rad=1, thr = 5,):
    
#     N = (2*rad+1)**2
#     padding = rad*1
#     nn_Unfold = torch.nn.Unfold(kernel_size=(2*rad+1,2*rad+1),dilation=1,padding=0,stride=1)
#     B,C,H,W = disp.shape
    
#     # unfold 
#     depth_padded = nn.functional.pad(depth, [rad,rad,rad,rad], mode='replicate')
#     disp_padded  = nn.functional.pad(disp, [rad,rad,rad,rad], mode='replicate')
#     depth_unfold = nn_Unfold(depth_padded)
#     depth_unfold = depth_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
#     disp_unfold  = nn_Unfold(disp_padded)
#     disp_unfold  = disp_unfold.view(B,C,N,H,W).permute(0,1,3,4,2) # [B,C,H,W,N]
    
    
#     # generate gradient
#     depth_gradient = depth_unfold - depth.unsqueeze(-1) # [B,C,H,W,N]
#     disp_gradient  = disp_unfold  -  disp.unsqueeze(-1) # [B,C,H,W,N]
    
#     without_margin = depth_gradient<2*thr
#     oversize_variation = (torch.abs(depth_gradient)/torch.abs(disp_gradient))
    
#     ############################################
#     ######### vote: True for against ###########
#     ############################################
#     # pixel pairs with enough depth variation but opposite disp gradient direction
#     direct_check = torch.zeros(depth_gradient.shape,device=depth_gradient.device).float() # [B,C,H,W,N]
#     direct_vote = torch.zeros(depth_gradient.shape,device=depth_gradient.device).bool() 
#     direct_check[depth_gradient > thr2] = 1 # [B,C,H,W,N]
#     direct_check[depth_gradient < -thr2] = -1 # [B,C,H,W,N]
#     direct_vote[direct_check * disp_gradient<0] = True # [B,C,H,W,N]
    
    
    
#     valid_mask = 0
#     disp[~valid_mask] = 0
    
#     return disp,valid_mask
