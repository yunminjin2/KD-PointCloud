"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020
"""

import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
from pointnet2 import pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn,bias=True):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True, groups=1):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points, fps_idx=None):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        if fps_idx is None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvDS(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvDS, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz_s, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz_s: input points position data for subsampling, [B, C, N]
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz_s = xyz_s.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz_s, self.npoint)
        new_xyz = index_points_gather(xyz_s, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class AdaptiveSampling(nn.Module):
    def __init__(self, nsample):
        super(AdaptiveSampling, self).__init__()
        self.nsample = nsample

    def forward(self, pc1_sparse, feat1_sparse, pc2_dense, feat2_dense):
        B = pc1_sparse.shape[0]
        N = pc1_sparse.shape[2]
        print(pc1_sparse.shape,feat1_sparse.shape,pc2_dense.shape,feat2_dense.shape)

        idx = knn_point(self.nsample, pc2_dense.permute(0, 2, 1), pc1_sparse.permute(0, 2, 1)) # [B, npoint, nsample]
        print(idx.shape)
        grouped_feat2 = index_points_group(feat2_dense.permute(0, 2, 1), idx).permute(0, 3, 1, 2) # [B, C, npoint, nsample]
        print(grouped_feat2.shape)
        sim = F.cosine_similarity(grouped_feat2, feat1_sparse.unsqueeze(-1), dim=1) # [B, npoint, nsample]
        print(sim.shape)
        sim_idx = torch.argmax(sim, dim=-1, keepdim=True) # [B, npoint, 1]
        print(sim_idx.shape)
        pc2_idx = torch.gather(idx, dim=-1, index=sim_idx).squeeze(-1).int()
        print(pc2_idx.shape)
        return pc2_idx


class PointConv4D(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv4D, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, c_xyz, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = c_xyz.shape[2]
        c_xyz = c_xyz.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_xyz = c_xyz

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class SetAbstractShuffle(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, npoint=None, bn=use_bn, use_leaky = True):
        super(SetAbstractShuffle, self).__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_convs = nn.ModuleList()
        self.pos_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel
        last_pos_channel = 3
        for i in range(len(mlp)-1):
            self.pos_convs.append(Conv1d(last_pos_channel, mlp[i], bn=bn, bias = False))
            self.mlp_convs.append(Conv1d(last_channel,  mlp[i], bn=bn, bias = False))
            last_channel =  mlp[i]
            last_pos_channel =  mlp[i]
        self.pos_convs.append(nn.Conv1d(last_pos_channel, mlp[-1], 1, bias = False))
        self.mlp_convs.append(nn.Conv1d(last_channel,  mlp[-1], 1, bias = False))
        self.bn2d = nn.BatchNorm2d(mlp[-1]) if bn else nn.Identity()
        # self.bias = nn.Parameter(torch.randn((1,  mlp[-1], 1, 1)), requires_grad=True)
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(Conv1d(last_channel, out_channel, bn=bn, bias=False))
                last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        new_points = points
        for conv in self.mlp_convs:
            new_points = conv(new_points)
        new_pos = xyz
        for conv in self.pos_convs:
            new_pos = conv(new_pos)

        for_group = torch.cat((new_points, new_pos), 1)
        xyz = xyz.permute(0, 2, 1)
        for_group = for_group.permute(0, 2, 1)

        if self.npoint is not None and self.npoint != xyz.size(1):
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_gather(xyz, fps_idx)
            new_pos = index_points_gather(new_pos.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
            new_points = index_points_gather(new_points.permute(0, 2, 1), fps_idx).permute(0, 2, 1)
        else: 
            new_xyz = xyz
        new_for_group, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, for_group)

        new_for_group = new_for_group.permute(0, 3, 1, 2)

        new_for_group = new_for_group[:,3:3+new_pos.size(1),...] + new_for_group[:,3+new_pos.size(1):,...] #+ self.bias
        new_points = self.relu(self.bn2d(new_for_group.contiguous() + new_points.unsqueeze(-1).contiguous() - new_pos.unsqueeze(-1).contiguous()))
        new_points = torch.max(new_points, -1)[0]

        for conv in self.mlp2_convs:
            new_points = conv(new_points)

        if self.npoint is not None and self.npoint != xyz.size(1):
            return new_xyz.permute(0, 2, 1), new_points, fps_idx
        else:    
            return new_points
        
class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost

class CrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayer,self).__init__()
        # self.fe1_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel,in_channel], pooling=pooling, corr_func=corr_func)
        # self.fe2_layer = FlowEmbedding(radius=radius, nsample=nsample, in_channel = in_channel, mlp=[in_channel, out_channel], pooling=pooling, corr_func=corr_func)
        # self.flow = nn.Conv1d(out_channel, 3, 1)

        self.nsample = nsample
        self.bn = bn
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()
        last_channel = in_channel  * 2 + 3
        for out_channel in mlp1:
            self.mlp1_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp1_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        if mlp2 is not None:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            last_channel = mlp1[-1] * 2 + 3
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(mlp_convs):
            if self.bn:
                bn = mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.mlp1_convs, self.mlp1_bns if self.bn else None)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossAttenLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossAttenLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.qk_conv1 = nn.Conv1d(in_channel, mlp1[-1], 1, bias=False)
        self.v_conv1 = Conv1d(in_channel, mlp1[0])
        self.v_pos1 = Conv1d(3, mlp1[0])
        fuse1 = []
        last_channel = mlp1[0]
        for i in range(1,len(mlp1)):
            fuse1.append(Conv1d(last_channel, mlp1[i]))
            last_channel = mlp1[i]
        self.fuse1 = nn.Sequential(*fuse1)
        self.beta1 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(-1)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        if mlp2 is not None:
            self.mlp2_convs = nn.ModuleList()
            if bn:
                self.mlp2_bns = nn.ModuleList()
            last_channel = mlp1[-1] * 2 + 3
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if bn:
                    self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        

    def cross(self, xyz1, xyz2, points1, points2, qk_conv, v_conv, v_pos, fuse, beta):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape

        qk1 = qk_conv(points1)  #B,C,N1
        qk2 = qk_conv(points2)  #B,C,N2
        v1 = v_conv(points1) 
        v2 = v_conv(points2) 
        v1_pos = v_pos(xyz1) 
        v2_pos = v_pos(xyz2) 

        energy12 = torch.bmm(qk1.permute(0,2,1), qk2) #B,N1,N2
        attention_12 = self.softmax(energy12)
        attention_21 = self.softmax(energy12.permute(0,2,1))

        out1 = torch.bmm(attention_12, (v2+v2_pos).permute(0,2,1)).permute(0,2,1) #B,C,N1
        out1 =  fuse(beta * out1 + v1 - v1_pos) #B,C,N1

        out2 = torch.bmm(attention_21, (v1+v1_pos).permute(0,2,1)).permute(0,2,1) #B,C,N2
        out2 =  fuse(beta* out2 + v2 - v2_pos)  #B,C,N2

        return out1, out2

    def flowembed(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(mlp_convs):
            if self.bn:
                bn = mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new, feat2_new = self.cross(pc1, pc2, feat1, feat2, self.qk_conv1, self.v_conv1, self.v_pos1, self.fuse1, self.beta1)
        feat1_final = self.flowembed(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightAtten(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightAtten,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossAtten(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossAtten,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel

        self.qk_conv = nn.Conv1d(in_channel, mlp2[-1], 1, bias=False)


    def forward(self, pc1, pc2, feat1, feat2):
        q = self.qk_conv(feat1)
        k = self.qk_conv(feat2)

        attn = q.transpose(-1, -2) @ k #B, N1, N2
        attn12 = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)
        attn21 = F.softmax(attn.transpose(-1, -2) / np.sqrt(q.size(-1)), dim=-2)

        feat1_new = feat2_new @ attn21
        feat2_new = feat1_new @ attn12

        return feat1_new, feat2_new

class CrossLayerLightAttentive(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightAttentive,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(10, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(10, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, cross1, cross2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1_cross = cross1(points1).permute(0, 2, 1)
        points2_cross = cross2(points2).permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2_cross, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1_cross.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)
        norm_xyz = torch.sqrt(torch.sum(torch.pow(direction_xyz, 2), dim=-1, keepdims=True))

        direction_xyz = torch.cat((neighbor_xyz, xyz1.view(B, N1, 1, C).repeat(1,1,self.nsample,1), direction_xyz, norm_xyz), -1)
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, D1+D2+3, nsample, N1

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        weight = F.softmax(new_points, -2)

        new_points = torch.sum(weight * index_points_group(points2, knn_idx).permute(0, 3, 2, 1), -2)
       

        return new_points


    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.cross_t1, self.cross_t2, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightAttentive2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightAttentive2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(10, mlp1[0], 1)
        self.pos11 = nn.Conv2d(10, mlp1[-1], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        self.trans1 = nn.Conv1d(in_channel, mlp1[-1], 1)
        self.trans2 = nn.Conv1d(in_channel, mlp1[-1], 1)
        self.bn11 = nn.BatchNorm2d(mlp1[-1]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(10, mlp2[0], 1)
            self.pos21 = nn.Conv2d(10, mlp1[-1], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()
            self.bn21 = nn.BatchNorm2d(mlp1[-1]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, cross1, cross2, pos, mlp, bn, pos2, bn2, trans1=None, trans2=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1_cross = cross1(points1).permute(0, 2, 1)
        points2_cross = cross2(points2).permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points1 = points1 if trans1 is None else trans1(points1)
        points2 = points2.permute(0, 2, 1) if trans2 is None else trans2(points2).permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2_cross, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1_cross.view(B, N1, 1, -1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)
        norm_xyz = torch.sqrt(torch.sum(torch.pow(direction_xyz, 2), dim=-1, keepdims=True))

        direction_xyz = torch.cat((neighbor_xyz, xyz1.view(B, N1, 1, C).repeat(1,1,self.nsample,1), direction_xyz, norm_xyz), -1)

        new_points = bn(grouped_points2 + grouped_points1 + pos(direction_xyz.permute(0, 3, 2, 1)))# B, D1+D2+3, nsample, N1

        # for i, conv in enumerate(mlp):
        #     new_points = conv(new_points)
        
        weight = F.tanh(new_points)

        new_points = torch.sum(weight * self.relu(bn(((index_points_group(points2, knn_idx)).permute(0, 3, 2, 1)+points1.unsqueeze(-2).repeat(1,1,self.nsample,1)+pos2(direction_xyz.permute(0, 3, 2, 1))))), -2)
       

        return new_points


    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1, self.pos11, self.bn11, self.trans1, self.trans2)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1, self.pos11, self.bn11, self.trans1, self.trans2)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.cross_t1, self.cross_t2, self.pos2, self.mlp2, self.bn2, self.pos21, self.bn21)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightAttentive3(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightAttentive3,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(10, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        self.trans1 = nn.Conv1d(in_channel, mlp1[-1], 1)

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points


    def attentive(self, xyz1, xyz2, points1, points2, cross1, cross2, pos, mlp, bn, trans=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1_cross = cross1(points1).permute(0, 2, 1)
        points2_cross = cross2(points2).permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1) if trans is None else trans(points2).permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2_cross, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1_cross.view(B, N1, 1, -1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)
        norm_xyz = torch.sqrt(torch.sum(torch.pow(direction_xyz, 2), dim=-1, keepdims=True))

        direction_xyz = torch.cat((neighbor_xyz, xyz1.view(B, N1, 1, C).repeat(1,1,self.nsample,1), direction_xyz, norm_xyz), -1)
        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, D1+D2+3, nsample, N1

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        weight = F.softmax(new_points, -2)

        new_points = torch.sum(weight * index_points_group(points2, knn_idx).permute(0, 3, 2, 1), -2)
       

        return new_points


    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.attentive(pc1, pc2, feat1, feat2, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1, self.trans1)
        feat2_new = self.attentive(pc2, pc1, feat2, feat1, self.cross_t11, self.cross_t22, self.pos1, self.mlp1, self.bn1, self.trans1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final
    # def forward(self, pc1, pc2, feat1, feat2, bid=False):

    #     feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
    #     feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

    #     if self.mlp2 is False:
    #         return feat1_new, feat2_new

    #     feat1_final = self.cross(pc1, pc2, self.cross_t1(feat1_new), self.cross_t2(feat2_new), self.pos2, self.mlp2, self.bn2)
    #     if bid:
    #         feat2_final = self.cross(pc2, pc1, self.cross_t1(feat2_new), self.cross_t2(feat1_new), self.pos2, self.mlp2, self.bn2)
    #         return feat1_new, feat2_new, feat1_final, feat2_final

    #     return feat1_new, feat2_new, feat1_final

class CrossLayerLightDouble(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2,  bn = use_bn, use_leaky = True):
        super(CrossLayerLightDouble,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        self.flow0 = SceneFlowEstimatorResidual(in_channel, mlp1[-1], weightnet = 4)
        self.warping = PointWarping()

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat0, flow0 = self.flow0(pc1, feat1, feat1_new)

        pc2_warp = self.warping(pc1, pc2, flow0)

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2_warp, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final, flow0, feat0

class CrossLayerLightVoteDouble(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2,  bn = use_bn, use_leaky = True):
        super(CrossLayerLightVoteDouble,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        self.flow0 = SceneFlowEstimatorResidual(in_channel, mlp1[-1], weightnet = 4)
        self.warping = PointWarping()

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))

            self.vote = nn.Conv2d(mlp2[-1], 1, 1)
            self.softmax = nn.Softmax(2)
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if return_vote:
            vote = self.softmax(self.vote(new_points)) # B, 1, K, N
            flow = torch.sum(vote * neighbor_xyz.permute(0, 3, 2, 1), 2) - xyz1.permute(0, 2, 1)
            return points_max, flow

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new,flow0 = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, return_vote=True)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        pc2_warp = self.warping(pc1, pc2, flow0)

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2_warp, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final, flow0



class CrossLayerLightS2D(nn.Module): #sparse to dense
    def __init__(self, nsample, in_channel, mlp1, mlp2, dense_channel= None, bn = use_bn, use_leaky = True):
        super(CrossLayerLightS2D,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            if dense_channel is not None:
                self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            else:
                self.cross_t2 = nn.Conv1d(mlp1[-1]+dense_channel, mlp2[0], 1)
                self.upsample = UpsampleFlow()

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2, pc2_d=None, feat2_d=None):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        if pc2_d is not None:
            feat2_new_d = torch.cat((feat2_d, self.upsample(pc2_d, pc2, feat2_new)))
            feat1_new = self.cross_t1(feat1_new)
            feat2_new = self.cross_t2(feat2_new_d)
            feat1_final = self.cross(pc1, pc2_d, feat1_new, feat2_new_d, self.pos2, self.mlp2, self.bn2)
        else:
            feat1_new = self.cross_t1(feat1_new)
            feat2_new = self.cross_t2(feat2_new)
            feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightVote(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightVote,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))

            self.vote = nn.Conv2d(mlp2[-1], 1, 1)
            self.softmax = nn.Softmax(2)
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if return_vote:
            vote = self.softmax(self.vote(new_points)) # B, 1, K, N
            flow = torch.sum(vote * neighbor_xyz.permute(0, 3, 2, 1), 2) - xyz1.permute(0, 2, 1)
            return torch.cat((points_max, flow), 1)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2, return_vote=True)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightVote1(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightVote1,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))

            self.vote = nn.Conv2d(mlp2[-1], 1, 1)
            self.softmax = nn.Softmax(2)
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if return_vote:
            vote = self.softmax(self.vote(new_points)) # B, 1, K, N
            flow = torch.sum(vote * neighbor_xyz.permute(0, 3, 2, 1), 2) - xyz1.permute(0, 2, 1)
            return points_max, flow

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new, flow = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, return_vote=True)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, torch.cat((feat1_final, flow), 1)


class CrossLayerLightVote2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightVote2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))

            self.vote = nn.Conv1d(mlp2[-1] + 3, 3, 1)
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, return_vote=False):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if return_vote:
            vote = self.vote(torch.cat((new_points, neighbor_xyz.permute(0, 3, 2, 1)),1)) # B, 1, K, N
            flow = torch.mean(vote, 2) - xyz1.permute(0, 2, 1)
            return torch.cat((points_max, flow), 1)

        return points_max

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2, return_vote=True)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightInterpolate(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightInterpolate,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2, pc1_d, pc2_d, feat1_d, feat2_d):

        feat1_new = self.cross(pc1, pc2_d, self.cross_t11(feat1), self.cross_t22(feat2_d), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1_d, self.cross_t11(feat2), self.cross_t22(feat1_d), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightAsym(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightAsym,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos11 = nn.Conv2d(3, mlp1[0], 1)
        self.pos21 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t21 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t12 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias11 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bias21 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn11 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        self.bn21 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t21(feat2), self.pos11, self.mlp1, self.bn11)
        feat2_new = self.cross(pc2, pc1, self.cross_t12(feat2), self.cross_t22(feat1), self.pos21, self.mlp1, self.bn21)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightOccout(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightOccout,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, occ1=None, occ2=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3


        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        if occ1 is not None:
            mask = occ1.unsqueeze(-2).repeat(1, 1, self.nsample, 1)
            new_points = new_points * mask
        if occ2 is not None:
            mask = index_points_group(occ2.permute(0,2,1), knn_idx).permute(0, 3, 2, 1)
            new_points = new_points * mask

        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2, occ1=None):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, occ1=None)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1, occ2=None)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2, occ1=occ1)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightOcc(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightOcc,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()


        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]

        self.occ = nn.Conv1d(mlp1[-1], 1, 1)

        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, occ1=None, occ2=None):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if occ1 is None  or occ2 is None:
            occ1 = torch.sigmoid(self.occ(feat1_new))
            occ2 = torch.sigmoid(self.occ(feat2_new))
        else:
            occ1 = torch.sigmoid(self.occ(feat1_new)+occ1)
            occ2 = torch.sigmoid(self.occ(feat2_new)+occ2)

        if self.mlp2 is False:
            return feat1_new, feat2_new, occ1, occ2

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new*occ1, feat2_new*occ2, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, occ1, occ2, feat1_final

class CrossLayerLightOcc2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightOcc2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()


        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]

        self.occ = nn.Conv1d(mlp1[-1], 1, 1)

        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, occ1=None, occ2=None):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if occ1 is None  or occ2 is None:
            occ1 = torch.sigmoid(self.occ(feat1_new))
            occ2 = torch.sigmoid(self.occ(feat2_new))
        else:
            occ1 = torch.sigmoid(self.occ(feat1_new)+occ1)
            occ2 = torch.sigmoid(self.occ(feat2_new)+occ2)

        if self.mlp2 is False:
            return feat1_new, feat2_new, occ1, occ2

        feat1_new = self.cross_t1(feat1_new)*occ1
        feat2_new = self.cross_t2(feat2_new)*occ2
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, occ1, occ2, feat1_final

class CrossLayerLightOcc3(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, occ_in=False, bn = use_bn, use_leaky = True):
        super(CrossLayerLightOcc3,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()


        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]

        if occ_in:
            self.occ = nn.Conv1d(mlp1[-1]+1, 1, 1)
        else:
            self.occ = nn.Conv1d(mlp1[-1], 1, 1)

        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, occ1=None, occ2=None):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if occ1 is None  or occ2 is None:
            occ1 = torch.sigmoid(self.occ(feat1_new))
            occ2 = torch.sigmoid(self.occ(feat2_new))
        else:
            occ1 = torch.sigmoid(self.occ(torch.cat((feat1_new,occ1), 1)))
            occ2 = torch.sigmoid(self.occ(torch.cat((feat2_new,occ2), 1)))

        if self.mlp2 is False:
            return feat1_new, feat2_new, occ1, occ2

        feat1_new = self.cross_t1(feat1_new)*occ1
        feat2_new = self.cross_t2(feat2_new)*occ2
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, occ1, occ2, feat1_final

class CrossLayerLightOcc4(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightOcc4,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()


        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]

        self.occ = nn.Conv1d(mlp1[-1], 1, 1)

        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        occ1 = torch.sigmoid(self.occ(feat1_new))
        occ2 = torch.sigmoid(self.occ(feat2_new))

        if self.mlp2 is False:
            return feat1_new, feat2_new, occ1, occ2

        feat1_new = self.cross_t1(feat1_new*occ1)
        feat2_new = self.cross_t2(feat2_new*occ2)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final


class CrossLayerLightSym(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightSym,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_final = self.cross(pc1, pc2, self.cross_t1(feat1_new), self.cross_t2(feat2_new), self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightSym2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightSym2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.lift1 = nn.Conv1d(mlp1[-1], mlp1[-1], 1)
            self.lift2 = nn.Conv1d(mlp1[-1], mlp1[-1], 1)
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new
        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final


class CrossLayerLight2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross(pc1, pc2, self.cross_t1(feat1_new), self.cross_t2(feat2_new), self.pos2, self.mlp2, self.bn2)
        feat2_new = self.cross(pc2, pc1, self.cross_t1(feat2_new), self.cross_t2(feat1_new), self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new

class CrossLayerLight3(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLight3,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1] *2, mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1] *2, mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(torch.cat((feat1_new,feat1_new),1))
        feat2_new = self.cross_t2(torch.cat((feat2_new,feat2_new),1))
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightGroup(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True, groups=1):
        super(CrossLayerLightGroup,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1, groups=groups)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1, groups=groups)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()
        self.groups = groups

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky, groups=groups))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1] *2, mlp2[0], 1, groups=groups)
            self.cross_t2 = nn.Conv1d(mlp1[-1] *2, mlp2[0], 1, groups=groups)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky, groups=groups))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3
        # new_points = new_points.view(B, self.groups, -1, self.nsample, N1).transpose(1,2).contiguous().view(B, -1, self.nsample, N1)
        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
            # new_points.view(B, self.groups, -1, self.nsample, N1).transpose(1,2).contiguous().view(B, -1, self.nsample, N1)

        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points
    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(torch.cat((feat1_new,feat1_new),1))
        feat2_new = self.cross_t2(torch.cat((feat2_new,feat2_new),1))
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final


class CrossLayerConvLight(nn.Module):
    def __init__(self, nsample, in_channel, out_channel1, out_channel2, weightnet = 8, bn = use_bn, use_leaky = True):
        super(CrossLayerConvLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.weightnet1 = WeightNet(3, weightnet)
        self.mlp1 = nn.ModuleList()

        # self.cross_t11 = nn.Conv1d(in_channel, out_channel1, 1)
        # self.cross_t22 = nn.Conv1d(in_channel, out_channel1, 1)
        # self.bias1 = nn.Parameter(torch.randn((1, out_channel1, 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm1d(out_channel1) if bn else nn.Identity()
        self.linear1 = nn.Linear(weightnet * in_channel * 2, out_channel1)

        self.out_channel2 = True if out_channel2 is not None else False

        if out_channel2 is not None:
            # self.cross_t1 = nn.Conv1d(out_channel1, out_channel2, 1)
            # self.cross_t2 = nn.Conv1d(out_channel1, out_channel2, 1)

            self.weightnet2 = WeightNet(3, weightnet)
            # self.bias2 = nn.Parameter(torch.randn((1, out_channel2, 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm1d(out_channel2) if bn else nn.Identity()
            self.linear2 = nn.Linear(weightnet * out_channel1 * 2, out_channel2)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, weightnet, linear, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        # direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        # new_points = grouped_points2 + grouped_points1# B, D1+D2, K, N
        new_points = torch.cat([grouped_points2, grouped_points1], 1)# B, D1+D2, K, N
        weights = weightnet(direction_xyz.permute(0, 3, 2, 1)) #B,W,K,N

        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, N1, -1)
        new_points = linear(new_points)
        new_points = self.relu(bn(new_points.permute(0, 2, 1)))

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        # feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.weightnet1, self.linear1, self.bn1)
        # feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.weightnet1, self.linear1, self.bn1)
        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.weightnet1, self.linear1, self.bn1)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.weightnet1, self.linear1, self.bn1)

        if self.out_channel2 is False:
            return feat1_new, feat2_new

        # feat1_new = self.cross_t1(feat1_new)
        # feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.weightnet2, self.linear2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerConvLight2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerConvLight2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = WeightNet(3, mlp1[-1])
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = WeightNet(3, mlp2[-1])
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))

        new_points = self.relu(bn(grouped_points2 + grouped_points1))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = torch.sum(direction_xyz * new_points, 2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerP2PConvLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerP2PConvLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.p2p11 = WeightNet(3, mlp1[-1])
        self.p2p12 = WeightNet(3, mlp1[-1])
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.p2p21 = WeightNet(3, mlp2[-1])
            self.p2p22 = WeightNet(3, mlp2[-1])
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, p2p1, p2p2, mlp, bn):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, C, nsample, N
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        new_points = self.relu(bn(grouped_points2 + grouped_points1 + pos(direction_xyz.permute(0, 3, 2, 1))))# B, C, nsample, N

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        weights = p2p1(direction_xyz.permute(0, 3, 2, 1))
        new_points = torch.sum(weights * new_points, 2)

        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        weights = p2p2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_new_points = index_points_group(new_points.permute(0, 2, 1), knn_idx) 
        new_points = torch.sum(weights * grouped_new_points.permute(0, 3, 2, 1), dim = 2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.p2p11, self.p2p12, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.p2p11, self.p2p12, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.p2p21, self.p2p22, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerP2PConvLight2(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerP2PConvLight2,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.p2p1 = WeightNet(3, mlp1[-1])
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.p2p2 = WeightNet(3, mlp2[-1])
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, p2p=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if p2p is not None:
            knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
            neighbor_xyz = index_points_group(xyz1, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            weights = p2p(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
            grouped_new_points = index_points_group(new_points.permute(0, 2, 1), knn_idx) 
            new_points = torch.sum(weights * grouped_new_points.permute(0, 3, 2, 1), dim = 2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1), self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2, self.p2p2)

        return feat1_new, feat2_new, feat1_final


class CrossLayerLightShift(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerLightShift,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
            last_channel = mlp1[i]
        self.weights1 = nn.Conv2d(mlp1[-1], 1, 1)
        self.softmax = nn.Softmax(-2)
        self.mlp2 = True if mlp2 is not None else False

        if mlp2 is not None:
            self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
            self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

            self.pos2 = nn.Conv2d(3, mlp2[0], 1)
            self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
            self.bn2 = nn.BatchNorm2d(mlp2[0]) if bn else nn.Identity()

            self.mlp2 = nn.ModuleList()
            for i in range(1, len(mlp2)):
                self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.upsample = UpsampleFlow()

    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn, weight=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz_pos = pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz_pos))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)

        new_points_max = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if weight is not None:
            w = self.softmax(weight(new_points))
            pc2_new = torch.sum(w*neighbor_xyz.permute(0, 3, 2, 1), -2)
            return pc2_new, new_points_max

        return new_points_max

    def forward(self, pc1, pc2, feat1, feat2):

        pc2_new, feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1, self.weights1)
        feat2_new = self.upsample(pc2_new, pc2, feat2)
        feat2_new = self.cross(pc2_new, pc1, self.cross_t11(feat2_new), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2_new, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None, neighr=3):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(neighr, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, neighr, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3

        # 3 nearest neightbor from dense around sparse & use 1/dist as the weights the same
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow 

class SceneFlowEstimatorResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow

class SceneFlowEstimatorResidualShuffle(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = ([128,128], [128,128]), mlp = [128, 128], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorResidualShuffle, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for ch_out in channels:
            pointconv = SetAbstractShuffle(neighbors, last_channel, ch_out, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out[-1] 
        
        self.mlp_convs = nn.ModuleList()
        for ch_out in mlp:
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow

class SceneFlowEstimatorResidualOcc(nn.Module):

    def __init__(self, feat_ch, cost_ch, occ_ch = 1, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidualOcc, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + occ_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        self.fc_occ = nn.Conv1d(last_channel, 1, 1)

    def forward(self, xyz, feats, cost_volume, flow = None, occ = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume], dim = 1)
        if occ is not None:
            new_points = torch.cat([new_points, occ], dim = 1)


        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        occ = self.fc_occ(new_points)
        
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow, occ
