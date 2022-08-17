
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from pointconv_util import Conv1d


class Bridge(torch.nn.Module):
    def __init__(self, feat_ch=512):
        super(Bridge, self).__init__()
        self.feat_ch = feat_ch

        self.mixed_layer = Conv1d(feat_ch + feat_ch, feat_ch)
        self.out_src = Conv1d(feat_ch + feat_ch, feat_ch)
        self.out_target = Conv1d(feat_ch + feat_ch, feat_ch)

    def forward(self, src_feat, target_feat):
        feats = torch.cat([src_feat, target_feat], dim=1)
        mixed = self.mixed_layer(feats)

        src_mixed = torch.cat([src_feat, mixed], dim=1)
        target_mixed = torch.cat([target_feat, mixed], dim=1)

        final_src_feat = self.out_src(src_mixed)
        final_tar_feat = self.out_src(target_mixed)

        return final_src_feat, final_tar_feat



from models_bid_lighttoken_res import PointConvBidirection
from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,8192,3)).float().cuda()
    input2 = torch.randn((1,512,256)).float().cuda()
    model = PointConvBidirection().cuda()
    bridge_model = Bridge().cuda()
    # print(model)
    pred_flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, feat1s, feat2s, crosses  = model(input,input,input,input)

    print(feat1s[0].size())

    bridge_model(feat1s[3],feat2s[3])

    macs, params = profile(bridge_model, inputs=(input2,input2))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    total = sum([param.nelement() for param in bridge_model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    for n,p in bridge_model.named_parameters():
        print(p.numel(), "\t", n, p.shape, )