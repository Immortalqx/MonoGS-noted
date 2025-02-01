#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


# L1 loss
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


# 计算加权L1损失，权重基于图像的边缘信息
# 然而，这个函数并没有使用。看起来MonoGS作者添加了这个函数，但可能没有很好的效果，最终论文没有提，开源的代码默认也没有使用
# TODO 这个函数有什么样的效果？先用l1_loss迭代一阵子，再用l1_loss_weight会不会更好？
def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


# L2 loss
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    # 生成一个一维高斯核，使用给定的窗口大小和标准差
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    # 对高斯核进行归一化，使其和为1
    return gauss / gauss.sum()


def create_window(window_size, channel):
    # 生成一个标准差为1.5的1D高斯窗
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 通过1D高斯窗与其转置相乘，生成2D高斯窗
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 扩展2D高斯窗以匹配图像的通道数
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


# 计算两张图片的结构相似度指数（SSIM）
def ssim(img1, img2, window_size=11, size_average=True):
    # 获取输入图像的通道数
    channel = img1.size(-3)
    # 创建用于SSIM计算的二维高斯窗
    window = create_window(window_size, channel)

    # 如果图像在CUDA设备上，将窗口移到相同的设备
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 使用创建的窗口计算SSIM
    return _ssim(img1, img2, window, window_size, channel, size_average)


# SSIM核心计算逻辑
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 计算局部均值（高斯滤波结果）
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)  # 分组卷积实现多通道处理
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # 计算均值平方和互乘项
    mu1_sq = mu1.pow(2)  # μ1²
    mu2_sq = mu2.pow(2)  # μ2²
    mu1_mu2 = mu1 * mu2  # μ1μ2

    # 计算方差和协方差
    sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )  # σ1² = E(X²)-μ1²
    sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )  # σ2² = E(Y²)-μ2²
    sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
    )  # σ12 = E(XY)-μ1μ2

    # SSIM计算常数（避免除以零）
    C1 = 0.01 ** 2  # 亮度稳定常数
    C2 = 0.03 ** 2  # 对比度稳定常数

    # 计算SSIM映射图
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # 根据参数决定返回维度
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
