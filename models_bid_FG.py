
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, CrossLayerLightFG as CrossLayer
from pointconv_util import SceneFlowEstimatorResidual
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time

scale = 1.0


class PointConvBidirection(nn.Module):
    def __init__(self, iters=2):
        super(PointConvBidirection, self).__init__()

        flow_nei = 32
        feat_nei = 32
        weightnet = 8
        self.scale = scale
        self.iters = iters
        #l0: 8192
        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet = weightnet)
        self.cross0 = CrossLayer(flow_nei, 32 + 32 , [32, 32], [32, 32])
        self.flow0 = SceneFlowEstimatorResidual(32 + 64, 32, weightnet = weightnet, channels = [64, 64], mlp = [64, 64])
        self.level0_1 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet = weightnet)
        self.cross1 = CrossLayer(flow_nei, 64 + 64, [64, 64], [64, 64])
        self.flow1 = SceneFlowEstimatorResidual(64 + 64, 64, weightnet = weightnet)
        self.level1_0 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet = weightnet)
        self.cross2 = CrossLayer(flow_nei, 128 + 128, [128, 128], [128, 128])
        self.flow2 = SceneFlowEstimatorResidual(128 + 64, 128, weightnet = weightnet)
        self.level2_0 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet = weightnet)
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256, 256, weightnet = weightnet)
        self.level3_0 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256, weightnet = weightnet)

        #deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N
        feat1_l0 = self.level0_lift(color1)
        feat1_l0 = self.level0(pc1_l0, feat1_l0)
        feat1_l0_1 = self.level0_1(feat1_l0)

        feat2_l0 = self.level0_lift(color2)
        feat2_l0 = self.level0(pc2_l0,feat2_l0)
        feat2_l0_1 = self.level0_1(feat2_l0)

        #l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        feat1_l1 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        feat2_l1 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1)

        #l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        feat1_l2 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        feat2_l2 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2)

        #l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        feat1_l3 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        feat2_l3 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3)

        #l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        #l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3, feat1_l3, feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cross3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        #l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        feat2_up = self.upsample(pc1_l2, pc1_l3, feat3)

        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        feat1_new_l2, feat2_new_l2, cross2 = self.cross2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2, feat1_l2, feat2_l2)

        new_feat1_l2 = torch.cat([feat1_l2, feat2_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cross2, up_flow2)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        #l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        feat1_up = self.upsample(pc1_l1, pc1_l2, feat2)

        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        feat1_new_l1, feat2_new_l1, cross1 = self.cross1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1, feat1_l1, feat2_l1)

        new_feat1_l1 = torch.cat([feat1_l1, feat1_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cross1, up_flow1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        feat0_up = self.upsample(pc1_l0, pc1_l1, feat1)

        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        feat1_new_l0, feat2_new_l0, cross0 = self.cross0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0, feat1_l0, feat2_l0)

        new_feat1_l0 = torch.cat([feat1_l0, feat0_up], dim = 1)
        feat0, flow0 = self.flow0(pc1_l0, new_feat1_l0, cross0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]
        feat1s = [feat1_l0_1, feat1_l1_2, feat1_l2_3, feat1_l3_4, feat1_l4, feat1_l3_2, feat1_l2_1, feat1_l1_0]
        feat2s = [feat2_l0_1, feat2_l1_2, feat2_l2_3, feat2_l3_4, feat2_l4,  feat2_l3_2, feat2_l2_1, feat2_l1_0]
        c_feat1s = [c_feat1_l0, c_feat1_l1, c_feat1_l2]
        c_feat2s = [c_feat2_l0, c_feat2_l1, c_feat2_l2]
        crosses = [cross0, cross1, cross2, cross3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2, feat1s, feat2s, c_feat1s, c_feat2s, crosses


def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    #generate GT list and mask1s
    gt_flows = [gt_flow]
    alphas = [alpha[0]]
    a = 0
    for i in range(1, len(fps_idxs)+1):
        fps_idx = fps_idxs[i - 1]
        if fps_idx is not None:
            sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
            gt_flows.append(sub_gt_flow)
            a += 1
            alphas.append(alpha[a])
        else:
            gt_flows.append(gt_flows[-1])
            alphas.append(alpha[a])

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i]
        total_loss += alphas[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss

def curvature(pc):
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def curvatureWarp(pc, warped_pc):
    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature

def multiScaleChamferSmoothCurvature(pc1, pc2, pred_flows):
    f_curvature = 0.3
    f_smoothness = 1.0
    f_chamfer = 1.0

    #num of scale
    num_scale = len(pred_flows)

    alpha = [0.02, 0.04, 0.08, 0.16]
    chamfer_loss = torch.zeros(1).cuda()
    smoothness_loss = torch.zeros(1).cuda()
    curvature_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        cur_pc1 = pc1[i] # B 3 N
        cur_pc2 = pc2[i]
        cur_flow = pred_flows[i] # B 3 N

        #compute curvature
        cur_pc2_curvature = curvature(cur_pc2)

        cur_pc1_warp = cur_pc1 + cur_flow
        dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
        moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

        chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

        #smoothness
        smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

        #curvature
        inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
        curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

        chamfer_loss += alpha[i] * chamferLoss
        smoothness_loss += alpha[i] * smoothnessLoss
        curvature_loss += alpha[i] * curvatureLoss

    total_loss = f_chamfer * chamfer_loss + f_curvature * curvature_loss + f_smoothness * smoothness_loss

    return total_loss, chamfer_loss, curvature_loss, smoothness_loss


from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,8192,3)).float().cuda()
    model = PointConvBidirection().cuda()
    # print(model)
    output = model(input,input,input,input)
    macs, params = profile(model, inputs=(input,input,input,input))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    # Check inference time
    timer = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(10):
        _ = model(input,input,input,input)
    with torch.no_grad():
        for i in range(100):
            start.record()
            _ = model(input,input,input,input)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            timer += start.elapsed_time(end)

    print(timer/100)