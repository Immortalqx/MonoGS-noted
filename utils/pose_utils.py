import numpy as np
import torch


def rt2mat(R, T):
    """将旋转矩阵和平移向量组合为齐次变换矩阵"""
    mat = np.eye(4)  # 创建4x4单位矩阵
    mat[0:3, 0:3] = R  # 填充旋转矩阵部分
    mat[0:3, 3] = T  # 填充平移向量部分
    return mat


def skew_sym_mat(x):
    """生成三维向量的反对称矩阵（李代数对应形式）"""
    device = x.device  # 获取输入张量设备信息
    dtype = x.dtype  # 获取输入张量数据类型
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)  # 初始化3x3零矩阵
    # 填充反对称矩阵元素
    ssm[0, 1] = -x[2]  # -z
    ssm[0, 2] = x[1]  # y
    ssm[1, 0] = x[2]  # z
    ssm[1, 2] = -x[0]  # -x
    ssm[2, 0] = -x[1]  # -y
    ssm[2, 1] = x[0]  # x
    return ssm


def SO3_exp(theta):
    """计算SO(3)指数映射（将李代数向量转换为旋转矩阵）
    $$ \exp([\theta]_\times) = \mathbf{I} + \frac{\sin\|\theta\|}{\|\theta\|}[\theta]_\times + \frac{1-\cos\|\theta\|}{\|\theta\|^2}[\theta]_\times^2 $$
    当角度趋近0时使用泰勒展开近似：
    $$ \exp([\theta]_\times) \approx \mathbf{I} + [\theta]_\times + \frac{1}{2}[θ]_\times^2 $$
    """
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)  # 生成反对称矩阵
    W2 = W @ W  # 计算反对称矩阵平方
    angle = torch.norm(theta)  # 计算旋转角度||θ||
    I = torch.eye(3, device=device, dtype=dtype)  # 3x3单位矩阵

    # 根据角度大小选择计算方式
    if angle < 1e-5:  # 小角度近似
        return I + W + 0.5 * W2
    else:  # 完整罗德里格斯公式
        return (
                I
                + (torch.sin(angle) / angle) * W  # 线性项
                + ((1 - torch.cos(angle)) / (angle ** 2)) * W2  # 二次项
        )


def V(theta):
    """计算SE(3)指数映射中的V矩阵
    $$ \mathbf{V} = \mathbf{I} + \frac{1-\cos\|\theta\|}{\|\theta\|^2}[\theta]_\times + \frac{\|\theta\| - \sin\|\theta\|}{\|\theta\|^3}[\theta]_\times^2 $$
    当角度趋近0时使用泰勒展开近似：
    $$ \mathbf{V} \approx \mathbf{I} + \frac{1}{2}[\theta]_\times + \frac{1}{6}[\theta]_\times^2 $$
    """
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)

    if angle < 1e-5:  # 小角度近似
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:  # 完整表达式
        V = (
                I
                + W * ((1.0 - torch.cos(angle)) / (angle ** 2))  # 线性项系数
                + W2 * ((angle - torch.sin(angle)) / (angle ** 3))  # 二次项系数
        )
    return V


def SE3_exp(tau):
    """计算SE(3)指数映射（将李代数向量转换为齐次变换矩阵）
    $$ \tau = \begin{bmatrix} \rho \\ \theta \end{bmatrix} $$
    $$ \exp(\tau) = \begin{bmatrix} \mathbf{R} & \mathbf{V}\rho \\ \mathbf{0} & 1 \end{bmatrix} $$
    """
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]  # 平移分量
    theta = tau[3:]  # 旋转分量

    R = SO3_exp(theta)  # 计算旋转矩阵
    t = V(theta) @ rho  # 计算平移向量：V(θ)ρ

    T = torch.eye(4, device=device, dtype=dtype)  # 创建4x4单位矩阵
    T[:3, :3] = R  # 填充旋转部分
    T[:3, 3] = t  # 填充平移部分
    return T


def update_pose(camera, converged_threshold=1e-4):
    """使用李代数更新相机位姿
    更新公式：$\mathbf{T}_{new} = \exp(\tau) \cdot \mathbf{T}_{old}$
    其中$\tau = [\rho; \theta]$为位姿增量
    """
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)  # 组合平移和旋转增量

    # 构建当前位姿矩阵
    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R  # 当前旋转
    T_w2c[0:3, 3] = camera.T  # 当前平移

    # 应用指数映射更新位姿
    new_w2c = SE3_exp(tau) @ T_w2c  # 矩阵相乘实现位姿组合

    # 提取新位姿参数
    new_R = new_w2c[0:3, 0:3]  # 新旋转矩阵
    new_T = new_w2c[0:3, 3]  # 新平移向量

    # 判断收敛条件：增量范数小于阈值
    converged = tau.norm() < converged_threshold

    # 更新相机参数并重置增量
    camera.update_RT(new_R, new_T)
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)

    return converged
