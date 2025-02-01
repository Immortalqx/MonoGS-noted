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

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


# 没有使用的函数
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


# 没有使用的函数
def PILtoTorch2(pil_image):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


# 本质上也是没有使用的函数
# 指数衰减学习率调度函数，支持延迟启用
def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    # def helper(step):
    #     if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
    #         # Disable this parameter
    #         return 0.0
    #     if lr_delay_steps > 0:
    #         # A kind of reverse cosine decay.
    #         delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
    #         )
    #     else:
    #         delay_rate = 1.0
    #     t = np.clip(step / max_steps, 0, 1)
    #     log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    #     return delay_rate * log_lerp

    return helper
    # return helper(lr_init=lr_init, lr_final=lr_final,
    #               lr_delay_steps=lr_delay_steps, lr_delay_mult=lr_delay_mult, max_steps=max_steps)


# 本质上就是上面def get_expon_lr_func的东西，搬出来了。所以MonoGS这里用的就是3DGS原来的策略
# 吐槽：莫名其妙的修改。。。
def helper(
        step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    # 如果步数小于0或学习率为0，直接返回0，表示不进行优化
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    # 如果设置了学习率延迟步数，计算延迟调整后的学习率
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    # 根据步数计算学习率的对数线性插值，实现从初始学习率到最终学习率的平滑过渡
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    # 返回调整后的学习率
    return delay_rate * log_lerp


# 提取下三角矩阵的元素
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


# 提取对称矩阵元素，调用 `strip_lowerdiag`
def strip_symmetric(sym):
    return strip_lowerdiag(sym)


# 根据四元数构建旋转矩阵，还是cuda上的tensor
def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# 根据四元数和尺度因子构建缩放-旋转矩阵，还是cuda上的tensor
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


# 并没有使用的函数
def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
