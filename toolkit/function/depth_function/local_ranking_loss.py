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
    
    depth_weight = Softshrink(depth_Gradient) # todo if use softshrink # [B,C,H,W,topK]
    
    disp_value = torch.log(1+torch.square(disp_Gradient)) # [B,C,H,W,topK]
    
    loss = Relu(-depth_weight*disp_direct*disp_value).sum(dim=-1) # [B,C,H,W]
    
    return loss


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
    direct_check = torch.zeros(depth_gradient.shape,device=depth_gradient.device).float() # [B,C,H,W,N]
    direct_vote = torch.zeros(depth_gradient.shape,device=depth_gradient.device).bool() 
    direct_check[depth_gradient > 0.67*thr] = 1 # [B,C,H,W,N]
    direct_check[depth_gradient < -0.67*thr] = -1 # [B,C,H,W,N]
    direct_vote[direct_check * disp_gradient<0] = True # [B,C,H,W,N]
    large_varia_vote = torch.logical_and(torch.abs(depth_gradient) > thr*2, torch.abs(disp_gradient) < 1) # [B,C,H,W,N]
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
    
    if return_score:
        return vote_score
    
    valid_mask = vote_score < Vote_ratio
    disp[~valid_mask] = 0
    
    return disp,valid_mask,vote_score
