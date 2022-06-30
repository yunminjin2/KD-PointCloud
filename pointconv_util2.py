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
from vn_layers import *

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
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

class PointConvFactor(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvFactor, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        # self.linear = nn.Linear(weightnet * in_channel, out_channel)
        self.w1 = Conv2d(in_channel*2, out_channel//2)
        self.w2 = Conv2d(64, 32)
        # if bn:
        #     self.bn_linear = nn.BatchNorm1d(out_channel)

        # self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


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
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, new_points.size(-1)*2, -1) #BxNxCxK * BxNxKxW => BxNxCxW
        # print(new_points.shape)
        new_points = self.w1(new_points.permute(0,2,1,3)) # BxC//8xNxW
        # print(new_points.shape)
        new_points = self.w2(new_points.view(B, new_points.size(1)//16, N, -1).permute(0,3,2,1)) # Bx16xNxC//16
        # print(new_points.shape)
        new_points = new_points.permute(0,1,3,2).reshape(B,-1,N)
        # print(new_points.shape)
        # new_points = self.linear(new_points)
        # if self.bn:
        #     new_points = self.bn_linear(new_points.permute(0, 2, 1))
        # else:
        #     new_points = new_points.permute(0, 2, 1)

        # new_points = self.relu(new_points)

        return new_points

class PointConvSVD(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvSVD, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Sequential(nn.Linear(weightnet * in_channel, out_channel//2),
                                    nn.Linear(out_channel//2, out_channel),)
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

class PointConvBias(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvBias, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(in_channel, out_channel)
        self.bias = nn.Parameter(torch.randn((1, 1, in_channel, weightnet)),requires_grad=True)
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
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1))+self.bias
        new_points = self.relu(new_points).permute(0,1,3,2) #BxNxWxK * BxNxKxC => BxNxWxC
        new_points = torch.sum(self.linear(new_points),-2)
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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

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

class PointConvSVDD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvSVDD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Sequential(nn.Linear(weightnet * in_channel, out_channel//2),
                                    nn.Linear(out_channel//2, out_channel),)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

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


class VNNConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, bn = use_bn, use_leaky = True):
        super(VNNConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample

        self.vn = VNLinearLeakyReLU(in_channel, out_channel)
        self.pool = VNMaxPool(out_channel)

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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # B, N, K, 3+C
        # grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        new_points = new_points.view(B, self.npoint, self.nsample, -1, 3).permute(0, 3, 4, 1, 2) # B, C, 3, N, K
        new_points = self.vn(new_points) # B, C, 3, N, K
        new_points = self.pool(new_points).view(B, -1, self.npoint)


        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvK(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvK, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.linear = nn.Linear(out_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points) # [B, npoint, nsample, C+D]

        # grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        kernel = self.kernel(new_points.permute(0, 3, 1, 2)) #B, C, N, S
        kernel = self.relu(kernel) # B, Out*In, N, S

        # B, N, Cout, S x B, N, S, Cin = B, N, Cout, Cin
        aggregation = torch.matmul(input = kernel.permute(0, 2, 1, 3), other = new_points)
        
        aggregation = self.relu(self.agg(aggregation.permute(0, 3, 1, 2))).squeeze(1)
        # B, N, S, C

        new_points = self.linear(aggregation)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvDRand(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvDRand, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_xyz = xyz[:, :self.npoint, :]

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

        return new_xyz.permute(0, 2, 1), new_points#, fps_idx

class SepConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(SepConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.agg = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.linear = nn.Linear(out_channel, out_channel)
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

        kernel = self.kernel(new_points.permute(0, 3, 1, 2))
        kernel = self.relu(kernel) # B, Out*In, N, S

        aggregation = torch.matmul(input = kernel.permute(0, 2, 1, 3), other = new_points.permute(0, 1, 2, 3))
        
        aggregation = self.relu(self.agg(aggregation.permute(0, 3, 1, 2))).squeeze(1)
        # B, N, S, C

        new_points = self.linear(aggregation)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvW(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, bn = use_bn, use_leaky = True):
        super(PointConvW, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.kernel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Conv1d(out_channel+nsample, out_channel+nsample, 1, bias=False),
            nn.BatchNorm1d(out_channel+nsample) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )

        self.weight_point = nn.Sequential(
            nn.Conv1d(nsample, nsample, 1, bias=False),
            nn.Sigmoid()
        )  

        self.weight_channel = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, 1, bias=False),
            nn.Sigmoid()
        )        

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
        C = points.shape[1]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # [B, npoint, nsample, C+D]
        new_points = self.kernel(new_points.permute(0, 3, 1, 2)) # B, Out, N, S

        channel_avg = torch.mean(new_points, -1) # B, C, N
        point_avg = torch.mean(new_points, 1) # B, N, S

        agg = torch.cat([channel_avg, point_avg.permute(0,2,1)], 1)# B, C+S, N

        agg = self.linear(agg)

        weight_point = self.weight_point(agg[:,-self.nsample:, ...]).permute(0,2,1).unsqueeze(1)# B, 1, N, S
        weight_channel = self.weight_channel(agg[:,:-self.nsample, ...]).unsqueeze(-1)# B, C, N, 1

        new_points = new_points * weight_channel * weight_point
        new_points = torch.mean(new_points, -1)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class LocalFeatureAggregation(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(LocalFeatureAggregation, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.mlp0 = Conv2d(in_channel, out_channel // 2, bn=True)
        self.lse_mlp1 = Conv2d(9, out_channel // 2, bn=True)
        self.score_fn1 = nn.Sequential(Conv2d(out_channel, out_channel, bias=False),nn.Softmax(dim=-1))
        self.mlp1 = Conv2d(out_channel, out_channel , bn=True)

        # self.lse_mlp2 = Conv2d(9, out_channel // 2, bn=True)
        # self.score_fn2 = nn.Sequential(Conv2d(out_channel, out_channel, bias=False),nn.Softmax(dim=-1))
        # self.mlp2 = Conv2d(out_channel, out_channel, bn=True)

        self.mlp = Conv1d(out_channel, out_channel)
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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        # new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(-2)
        grouped_points = index_points_group(points, idx) # [B, npoint, nsample, C]
        
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2) 
        grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2) 
        new_points = self.mlp0(grouped_points.permute(0, 3, 1, 2))
        extended_xyz = new_xyz.permute(0, 2, 1).unsqueeze(-1).expand(B, 3, self.npoint, self.nsample)

        # Local Spatial Encoding #1
        concat = torch.cat((extended_xyz, grouped_xyz, grouped_xyz_norm), dim=-3)
        lse1 = torch.cat((self.lse_mlp1(concat), new_points.expand(B, -1, self.npoint, self.nsample)), dim=-3) # B, C, N, S
        # Attentive Pooling #1
        scores1 = self.score_fn1(lse1) # B, C // 2, N, S
        features1 = torch.sum(scores1 * lse1, dim=-1, keepdim=True) # B, C // 2, N, S
        new_points = self.mlp1(features1).squeeze(-1) # B, N, C // 2

        # features1 = new_points.permute(0, 2, 1)
        # # Local Spatial Encoding #2
        # grouped_points2 = index_points_group(features1, idx).permute(0, 3, 1, 2)
        # lse2 = torch.cat((self.lse_mlp2(concat), grouped_points2.expand(B, -1, self.npoint, self.nsample)), dim=-3) # B, C, N, S
        # # Attentive Pooling #2
        # scores2 = self.score_fn2(lse2)
        # features2 = torch.sum(scores2 * lse2, dim=-1, keepdim=True) # B, C, N, S
        # features2 = self.mlp2(features2).squeeze(-1) # B, N, C

        # new_points = self.mlp(features2)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class SetAbstract(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(SetAbstract, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()

        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
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
        xyz = xyz.permute(0, 2, 1)
        points = F.conv1d(points, self.mlp_convs[0].weight[:,3:,...].squeeze(-1))
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, xyz, points)
        grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2)
        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = F.conv2d(grouped_xyz_norm, self.mlp_convs[0].weight[:,:3,...])+ new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(new_points))
            else:
                new_points = self.relu(new_points)

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_points

class SetAbstractD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(SetAbstractD, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
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
        xyz = xyz.permute(0, 2, 1)
        points = F.conv1d(points, self.mlp_convs[0].weight[:,3:,...].squeeze(-1))
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2)
        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = F.conv2d(grouped_xyz_norm, self.mlp_convs[0].weight[:,:3,...])+ new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(new_points))
            else:
                new_points = self.relu(new_points)

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class SetAbstractFuse(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(SetAbstractFuse, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.weight = nn.Sequential(
            nn.Conv2d(3, nsample, 1, bias=False),
            nn.BatchNorm2d(nsample),
            nn.Softmax(-1),
        )
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
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
        xyz = xyz.permute(0, 2, 1)
        points = F.conv1d(points, self.mlp_convs[0].weight.squeeze(-1))
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, xyz, points) # [B, npoint, nsample, C+D]
        grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2) # [B, 3, N, K]
        new_points = new_points.permute(0, 3, 1, 2) # [B, C, N, K]

        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = torch.einsum("bjnk,bcnk->bcnj",[self.weight(grouped_xyz_norm),new_points[:,3:,...]]) + new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(new_points))
            else:
                new_points = self.relu(new_points)

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_points

class SetAbstractFuseD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(SetAbstractFuseD, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.weight = nn.Sequential(
            nn.Conv2d(3, nsample, 1, bias=False),
            nn.BatchNorm2d(nsample),
            nn.Softmax(-1),
        )
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
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
        xyz = xyz.permute(0, 2, 1)
        points = F.conv1d(points, self.mlp_convs[0].weight.squeeze(-1))
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz_norm = grouped_xyz_norm.permute(0, 3, 1, 2)
        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            if i == 0:
                new_points = torch.einsum("bjnk,bcnk->bcnj",[self.weight(grouped_xyz_norm),new_points[:,3:,...]]) + new_points[:,3:,...]
            else:
                new_points = conv(new_points)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(new_points))
            else:
                new_points = self.relu(new_points)

        new_points = torch.max(new_points, -1)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = self.relu(conv(new_points))

        return new_xyz.permute(0, 2, 1), new_points, fps_idx


class PointAtten(nn.Module):
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
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points) # B, N, S, C

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) # B, 16, S, N

        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1) # B, N, C, S x B, N, S, 16

        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

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

class NoCrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, bn = use_bn, use_leaky = True, output_clue=False):
        super(NoCrossLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.output_clue = output_clue
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()
        last_channel = in_channel  * 2 + 3
        for out_channel in mlp1:
            self.mlp1_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp1_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, mlp_convs, mlp_bns, output_clue=False):
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
        
        max_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        if output_clue:
            mask = (new_points == torch.max(new_points, -2, keepdim=True)[0]).to(dtype=torch.float32)
            masked_points = torch.mul(mask, new_points)
            return max_points, torch.sum(masked_points, 1), knn_idx

        return max_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.mlp1_convs, self.mlp1_bns if self.bn else None, output_clue = self.output_clue)

        return feat1_new

class NoCrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, bn = use_bn, use_leaky = True, output_clue=False):
        super(NoCrossLayerLight,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.output_clue = output_clue
        self.mlp1_convs = nn.ModuleList()
        if bn:
            self.mlp1_bns = nn.ModuleList()

        self.cross_t1 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t2 = nn.Conv1d(in_channel, mlp1[0], 1)

        self.pos = nn.Conv2d(3, mlp1[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        self.mlp = nn.ModuleList()
        for i in range(1, len(mlp1)):
            self.mlp.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
        
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

        feat1_new = self.cross(pc1, pc2, self.cross_t1(feat1),  self.cross_t2(feat2), self.pos, self.mlp, self.bn)

        return feat1_new

class CrossConvLayer(nn.Module):
    def __init__(self, nsample, in_channel, mid_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(CrossConvLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn

        self.weightnet1 = WeightNet(3, weightnet)
        self.linear1 = nn.Linear(weightnet * in_channel*2, mid_channel)
        if bn:
            self.bn_linear1 = nn.BatchNorm1d(mid_channel)

        self.cross2 = True if out_channel is not None else False
        if out_channel is not None:
            self.weightnet2 = WeightNet(3, weightnet)
            self.linear2 = nn.Linear(weightnet * mid_channel*2, out_channel)
            if bn:
                self.bn_linear2 = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, weightnet, linear, bn_linear):
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
        grouped_xyz = direction_xyz.permute(0, 3, 2, 1)

        new_points = torch.cat([grouped_points1, grouped_points2], dim = -1) # B, N1, nsample, D1+D2
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2, nsample, N1]
        weights = weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, N1, -1)
        new_points = linear(new_points)
        if self.bn:
            new_points = bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, feat1, feat2, self.weightnet1, self.linear1, self.bn_linear1 if self.bn else None)
        feat2_new = self.cross(pc2, pc1, feat2, feat1, self.weightnet1, self.linear1, self.bn_linear1 if self.bn else None)
        if self.cross2 is False:
            return feat1_new, feat2_new
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.weightnet2, self.linear2, self.bn_linear2 if self.bn else None)

        return feat1_new, feat2_new, feat1_final


class CrossLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayer,self).__init__()

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
        
        self.mlp2 = True if mlp2 is not None else False
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
        if self.mlp2 is False:
            return feat1_new, feat2_new
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class FlowEmbeddingLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(FlowEmbeddingLayer,self).__init__()

        self.nsample = nsample
        self.mlp = nn.ModuleList()
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        self.bn = nn.BatchNorm2d(mlp[0]) if bn else nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], bn=bn, use_leaky=use_leaky))
        
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points1, points2 = self.t11(points1), self.t22(points2)
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        grouped_points2 = index_points_group(points2, knn_idx).permute(0, 3, 2, 1) # B, D2, nsample, N1
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1).permute(0, 3, 2, 1)

        direction_xyz = self.pos(direction_xyz.permute(0, 3, 2, 1))
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz))# B, N1, nsample, D1+D2+3

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

class CrossPoolLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossPoolLayer,self).__init__()

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
        
        self.mlp2 = True if mlp2 is not None else False
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
        if self.mlp2 is False:
            return feat1_new, feat2_new
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.mlp2_convs, self.mlp2_bns if self.bn else None)

        return feat1_new, feat2_new, feat1_final

class CrossTransLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(CrossTransLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn

        self.cross_convs = nn.ModuleList()
        self.qk_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.qk_convs.append(Conv1d(last_channel, last_channel, bn=bn, use_leaky=use_leaky))
            self.cross_convs.append(Conv1d(last_channel+3, out_channel, bn=bn, use_leaky=use_leaky))
            last_channel = out_channel
        
        self.fe = None
        if mlp2 is not None:
            self.fe = FlowEmbeddingLayer(nsample, last_channel, mlp2)
        
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, cross_conv, qk_conv):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape

        points1_qk = qk_conv(points1) # B, D1, N1
        points2_qk = qk_conv(points2) # B, D2, N2

        atten = torch.matmul(points1_qk.transpose(1,2), points2_qk) # B, N1, N2

        atten1 = self.softmax(atten.transpose(1,2)) # B, N2, N1
        atten2 = self.softmax(atten) # B, N1, N2

        new_points1 = cross_conv(torch.matmul(torch.cat([xyz2, points2],1),atten1)) + points1
        new_points2 = cross_conv(torch.matmul(torch.cat([xyz1, points1],1),atten2)) + points2

        return new_points1, new_points2

    def forward(self, pc1, pc2, feat1, feat2):
        
        feat1_new, feat2_new = feat1, feat2
        for i in range(len(self.cross_convs)):
            feat1_new, feat2_new = self.cross(pc1, pc2, feat1_new, feat2_new, self.cross_convs[i], self.qk_convs[i])
        if self.fe is not None:
            feat1_final = self.fe(pc1, pc2, feat1_new, feat2_new)
        return feat1_new, feat2_new, feat1_final

class CrossLocalTransLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, mlp2=None, bn = use_bn, use_leaky = True):
        super(CrossLocalTransLayer,self).__init__()

        self.nsample = nsample
        self.bn = bn

        self.cross_convs = nn.ModuleList()
        self.qk_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.qk_convs.append(Conv1d(last_channel, last_channel, bn=bn, use_leaky=use_leaky))
            self.cross_convs.append(Conv1d(last_channel+3, out_channel, bn=bn, use_leaky=use_leaky))
            last_channel = out_channel
        
        self.fe = None
        if mlp2 is not None:
            self.fe = FlowEmbeddingLayer(nsample, last_channel, mlp2)
        
        self.softmax = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, cross_conv, qk_conv):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape

        points1_qk = qk_conv(points1) # B, D1, N1
        points2_qk = qk_conv(points2) # B, D2, N2

        xyz1_t = xyz1.permute(0, 2, 1)
        xyz2_t = xyz2.permute(0, 2, 1)
        points1_qkt = points1_qk.permute(0, 2, 1) # B, N1, D1
        points2_qkt = points2_qk.permute(0, 2, 1) # B, N2, D2

        knn_idx2 = knn_point(self.nsample*2, xyz2_t, xyz1_t) # B, N1, nsample
        grouped_xyz2 = index_points_group(xyz2_t, knn_idx2).permute(0, 1, 3, 2)
        direction_xyz2 = grouped_xyz2 - xyz1_t.unsqueeze(-1) # B, N1, C, K
        grouped_points2_qkt = index_points_group(points2_qkt, knn_idx2) # B, N1, K, D2
        grouped_points2_t = index_points_group(points2.permute(0, 2, 1), knn_idx2).permute(0, 1, 3, 2) # B, N1, D2, K
        atten2 = torch.matmul(grouped_points2_qkt, points1_qkt.unsqueeze(-1)) # B, N1, K, 1
        atten2 = self.softmax(atten2) # B, N1, K, 1
        points2_v = torch.matmul(torch.cat([direction_xyz2, grouped_points2_t],-2), atten2).transpose(1,2) # B, 3+D2, N1, 1

        knn_idx1= knn_point(self.nsample*2, xyz1_t, xyz2_t)
        grouped_xyz1 = index_points_group(xyz1_t, knn_idx1).permute(0, 1, 3, 2)
        direction_xyz1 = grouped_xyz1 - xyz2_t.unsqueeze(-1)
        grouped_points1_qkt = index_points_group(points1_qkt, knn_idx1)
        grouped_points1_t = index_points_group(points1.permute(0, 2, 1), knn_idx1).permute(0, 1, 3, 2)
        atten1 = torch.matmul(grouped_points1_qkt, points2_qkt.unsqueeze(-1))
        atten1 = self.softmax(atten1)
        points1_v = torch.matmul(torch.cat([direction_xyz1, grouped_points1_t],-2), atten1).transpose(1,2)

        new_points1 = cross_conv(points2_v.squeeze(-1)) + points1
        new_points2 = cross_conv(points1_v.squeeze(-1)) + points2

        return new_points1, new_points2

    def forward(self, pc1, pc2, feat1, feat2):
        
        feat1_new, feat2_new = feat1, feat2
        for i in range(len(self.cross_convs)):
            feat1_new, feat2_new = self.cross(pc1, pc2, feat1_new, feat2_new, self.cross_convs[i], self.qk_convs[i])
        if self.fe is not None:
            feat1_final = self.fe(pc1, pc2, feat1_new, feat2_new)
        return feat1_new, feat2_new, feat1_final

class CrossLayerPoolLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, bn = use_bn, use_leaky = True):
        super(CrossLayerPoolLight,self).__init__()

        self.nsample = nsample
        self.bn = bn

        self.pos1 = nn.ModuleList()
        self.cross1_1 = nn.ModuleList()
        self.cross1_2 = nn.ModuleList()
        self.bias1 = nn.ParameterList()
        self.bn1 = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp1:
            self.pos1.append(nn.Conv2d(3,out_channel, 1))
            self.cross1_1.append(nn.Conv1d(last_channel, out_channel, 1))
            self.cross1_2.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bias1.append(nn.Parameter(torch.randn((1, out_channel, 1, 1)),requires_grad=True))
            self.bn1.append(nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity())
            last_channel = out_channel
        
        self.mlp2 = True if mlp2 is not None else False
        if mlp2 is not None:
            self.pos2 = nn.ModuleList()
            self.cross2_1 = nn.ModuleList()
            self.cross2_2 = nn.ModuleList()
            self.bias2 = nn.ParameterList()
            self.bn2 = nn.ModuleList()

            for out_channel in mlp2:
                self.pos2.append(nn.Conv2d(3,out_channel, 1))
                self.cross2_1.append(nn.Conv1d(last_channel, out_channel, 1))
                self.cross2_2.append(nn.Conv1d(last_channel, out_channel, 1))
                self.bias2.append(nn.Parameter(torch.randn((1, out_channel, 1, 1)),requires_grad=True))
                self.bn2.append(nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity())
                last_channel = out_channel
            
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def cross(self, xyz1, xyz2, points1, points2, pos, bias, bn):
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
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz+bias))# B, N1, nsample, D1+D2+3
        
        new_points = F.max_pool2d(new_points, (new_points.size(2), 1)).squeeze(2)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):

        feat1_new, feat2_new = feat1, feat2
        for i in range(len(self.cross1_1)):
            feat1_new_ = self.cross(pc1, pc2, self.cross1_1[i](feat1_new), self.cross1_2[i](feat2_new), self.pos1[i], self.bias1[i], self.bn1[i])
            feat2_new_ = self.cross(pc2, pc1, self.cross1_1[i](feat2_new), self.cross1_2[i](feat1_new), self.pos1[i], self.bias1[i], self.bn1[i])
            feat1_new, feat2_new = feat1_new_, feat2_new_

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_final = feat1_new
        for i in range(len(self.cross2_1)):
            feat1_final = self.cross(pc1, pc2, self.cross2_1[i](feat1_final), self.cross2_2[i](feat2_new), self.pos2[i], self.bias2[i], self.bn2[i])

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
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)

        feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final

class CrossLayerLightUp(nn.Module):
    def __init__(self, nsample, in_channel_dense, in_channel_sparse, mlp1, mlp2=None, bn = use_bn, use_leaky = True):
        super(CrossLayerLightUp,self).__init__()

        self.nsample = nsample
        self.bn = bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        self.cross_t11 = nn.Conv1d(in_channel_dense, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel_sparse, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        self.bn1 = nn.BatchNorm2d(mlp1[0]) if bn else nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], bn=bn, use_leaky=use_leaky))
        
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

    def forward(self, pc1, pc2, feat1, feat2, pc1_s=None, pc2_s=None, feat1_s=None, feat2_s=None):
        # _, feat1_new = self.fe1_layer(pc1, pc2, feat1, feat2)
        # _, feat2_new = self.fe1_layer(pc2, pc1, feat2, feat1)
        # _, feat1_final = self.fe2_layer(pc1, pc2, feat1_new, feat2_new)
        # flow1 = self.flow(feat1_final)
        if  pc1_s is None or  pc2_s is None or feat1_s is None or feat2_s is None:
            pc1_s, pc2_s, feat1_s, feat2_s = pc1, pc2, feat1, feat2

        feat1_new = self.cross(pc1, pc2_s, self.cross_t11(feat1), self.cross_t22(feat2_s), self.pos1, self.mlp1, self.bn1)
        feat2_new = self.cross(pc2, pc1_s, self.cross_t11(feat2), self.cross_t22(feat1_s), self.pos1, self.mlp1, self.bn1)

        if self.mlp2 is False:
            return feat1_new, feat2_new

        feat1_new = self.cross_t1(feat1_new)
        feat2_new = self.cross_t2(feat2_new)
        feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

        return feat1_new, feat2_new, feat1_final


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

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
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
        knn_idx = knn_point(3, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
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

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
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
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

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

class SceneFlowEstimatorSepResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorSepResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = SepConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
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

class SceneFlowEstimatorResidualSmooth(nn.Module):

    def __init__(self, feat_ch, bid_ch, cost_ch, flow_ch = 3, mlp = [256, 128], channels = [128], neighbors = 16, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidualSmooth, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + bid_ch + cost_ch

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out
        # self.fc1 = nn.Conv1d(last_channel, 3, 1)


        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, bid, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        new_points = torch.cat([feats, cost_volume, bid], dim = 1)

        for conv in self.mlp_convs:
            new_points = conv(new_points)
        # flow_local = self.fc1(new_points)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)
        flow_local = self.fc(new_points)

        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return new_points, flow

class SceneFlowEstimatorResidualBias(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidualBias, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConvBias(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
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

class SceneFlowEstimatorSetconvResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = ([128,128], [128,128]), mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorSetconvResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = SetAbstract(neighbors, last_channel, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out[-1] 
        
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

class SceneFlowEstimatorSetconvFuseResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = ([128,128], [128,128]), mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorSetconvFuseResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = SetAbstractFuse(neighbors, last_channel, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out[-1] 
        
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

class SceneFlowEstimatorResidualFactor(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidualFactor, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConvK(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
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

class SceneFlowEstimatorResidualSVD(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True, weightnet=16):
        super(SceneFlowEstimatorResidualSVD, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConvSVD(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True,weightnet=weightnet)
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
