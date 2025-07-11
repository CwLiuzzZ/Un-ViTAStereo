import torch
import torch.nn as nn
import torch.nn.functional as F

from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv

class JBULearnedRange(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, output_dim, midd_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        # self.down_channel1 = nn.Sequential(nn.Conv2d(in_channels=feat_dim,out_channels=midd_dim,kernel_size=1, stride=1, padding=0))
        # self.down_channel2 = nn.Sequential(nn.Conv2d(in_channels=midd_dim,out_channels=output_dim,kernel_size=1, stride=1, padding=0))
        self.down_channel1 = nn.Conv2d(in_channels=feat_dim,out_channels=midd_dim,kernel_size=1, stride=1, padding=0)
        self.down_channel2 = nn.Conv2d(in_channels=midd_dim,out_channels=output_dim,kernel_size=1, stride=1, padding=0)

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
                
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape # 2 3 72 120
        SB, SC, SH, SQ = source.shape # 2 1024 36 60
        assert (SB == GB)
        
        spatial_kernel = self.get_spatial_kernel(source.device) # [1,49,1,1] for window size 
        range_kernel = self.get_range_kernel(guidance) # [2,49,GH,GW] for window size 

        combined_kernel = range_kernel * spatial_kernel # [2,49,GH,GW]
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7) # [2,49,GH,GW]

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1)) # [2,49,GH,GW]
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter) # [2,GH,GW,7,7] 7 for diameter

        source = self.down_channel1(source)
        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source) # [2,midd_dim,GH,GW]
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect') # [2,midd_dim,GH+7,GW+7]

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW
        result =  AdaptiveConv.apply(hr_source_padded, combined_kernel) # [2,midd_dim,GH,GW]
        result = self.down_channel2(result)
        return result


class JBUStack(torch.nn.Module):

    def __init__(self, ViTAS_channel,out_channels,midd_channels,feat_dim=32):
        print(ViTAS_channel,out_channels,midd_channels)
        super().__init__()
        self.up1 = JBULearnedRange(3, ViTAS_channel, out_channels[1], midd_channels[1], 32, radius=3)
        self.up2 = JBULearnedRange(3, out_channels[1], out_channels[0], midd_channels[0], 32, radius=3)
        self.fixup_proj1 = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1))
        self.fixup_proj2 = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1))

        self.act_3_preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=ViTAS_channel,
                out_channels=out_channels[2],
                kernel_size=1, stride=1, padding=0,
            )
        )
        
        self.act_4_preprocess = nn.Sequential(
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

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled
    
    def act_2_preprocess(self, source, guidance):
        source_1 = self.upsample(source, guidance, self.up1)
        # return source_1
        return self.fixup_proj1(source_1) * 0.1 + source_1

    def act_1_preprocess(self, source, guidance):
        source_1 = self.upsample(source, guidance, self.up1)
        source_2 = self.upsample(source_1, guidance, self.up2)
        # return source_2
        return self.fixup_proj2(source_2) * 0.1 + source_2

