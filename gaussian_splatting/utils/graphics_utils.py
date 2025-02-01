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

import math
from typing import NamedTuple

import numpy as np
import torch


# 定义一个基本点云的数据结构，包含点、颜色和法线
# 主要用来保存新增的RGB-D点云
class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


# 构建从世界坐标系到相机坐标系的变换矩阵，并没有使用
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


# 构建从世界坐标系到相机坐标系的变换矩阵，并支持额外的平移和缩放操作，而且还是tensor的操作
# 似乎MonoGS中并没有用到external translate or scale
def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # 看起来，MonoGS的作者也被各种各样的位姿折磨过。
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


# 根据视锥体参数生成投影矩阵
# 没有用到的函数
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))  # 计算垂直视场角的一半的正切值
    tanHalfFovX = math.tan((fovX / 2))  # 计算水平视场角的一半的正切值

    top = tanHalfFovY * znear  # 计算视锥体顶面位置
    bottom = -top  # 计算底面位置
    right = tanHalfFovX * znear  # 计算右侧位置
    left = -right  # 计算左侧位置

    P = torch.zeros(4, 4)  # 初始化4x4投影矩阵

    z_sign = 1.0  # 深度值的正负方向

    P[0, 0] = 2.0 * znear / (right - left)  # 设置投影矩阵的横向缩放因子
    P[1, 1] = 2.0 * znear / (top - bottom)  # 设置纵向缩放因子
    P[0, 2] = (right + left) / (right - left)  # 设置横向偏移
    P[1, 2] = (top + bottom) / (top - bottom)  # 设置纵向偏移
    P[3, 2] = z_sign  # 设置深度方向
    P[2, 2] = -(zfar + znear) / (zfar - znear)  # 设置深度缩放因子
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)  # 设置深度偏移
    return P


# 生成投影矩阵，考虑图像传感器的光心和焦距
def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0  # 计算左边界
    right = ((2 * cx - W) / W + 1.0) * W / 2.0  # 计算右边界
    top = ((2 * cy - H) / H + 1.0) * H / 2.0  # 计算顶部
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0  # 计算底部
    left = znear / fx * left  # 转换到近平面
    right = znear / fx * right  # 转换到近平面
    top = znear / fy * top  # 转换到近平面
    bottom = znear / fy * bottom  # 转换到近平面
    P = torch.zeros(4, 4)  # 初始化4x4投影矩阵

    z_sign = 1.0  # 深度值正负方向

    P[0, 0] = 2.0 * znear / (right - left)  # 横向缩放
    P[1, 1] = 2.0 * znear / (top - bottom)  # 纵向缩放
    P[0, 2] = (right + left) / (right - left)  # 横向偏移
    P[1, 2] = (top + bottom) / (top - bottom)  # 纵向偏移
    P[3, 2] = z_sign  # 设置深度方向
    P[2, 2] = z_sign * zfar / (zfar - znear)  # 深度缩放
    P[2, 3] = -(zfar * znear) / (zfar - znear)  # 深度偏移

    return P


# 根据视场角和像素大小计算焦距
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


# 根据焦距和像素大小计算视场角
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
