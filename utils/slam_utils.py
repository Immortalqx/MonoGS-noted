import torch


# 使用Scharr滤波器计算图像的水平和垂直梯度
def image_gradient(image):
    # Compute image gradient using Scharr Filter

    # 获取输入图像的通道数
    c = image.shape[0]

    # 定义Scharr滤波器的垂直和水平卷积核
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]],  # 垂直方向梯度检测核
        dtype=torch.float32,
        device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],  # 水平方向梯度检测核
        dtype=torch.float32,
        device="cuda"
    )

    # 计算归一化系数（Scharr滤波器的绝对值和）
    normalizer = 1.0 / torch.abs(conv_y).sum()  # 1/32

    # 对输入图像进行反射填充（上下左右各填充1像素）
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]  # 添加batch维度

    # 计算水平梯度（使用垂直卷积核）
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img,
        conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1),  # 扩展卷积核维度 [C, 1, 3, 3]
        groups=c  # 分组卷积处理多通道
    )

    # 计算垂直梯度（使用水平卷积核）
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img,
        conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1),
        groups=c
    )

    # 移除临时添加的batch维度并返回结果
    return img_grad_v[0], img_grad_h[0]


# 生成图像梯度有效区域掩码
def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]  # 输入通道数

    # 创建全1的3x3卷积核（用于检测有效区域）
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")

    # 反射填充并创建二进制掩码（绝对值超过阈值的区域）
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps  # 转换为布尔张量

    # 卷积计算有效区域（将True视为1，False视为0）
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    # 生成掩码（卷积结果等于核元素和表示全部邻域有效）
    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


# 深度图正则化损失计算
def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    # 获取深度图的有效区域掩码
    mask_v, mask_h = image_gradient_mask(depth)

    # 计算灰度图的梯度
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))  # RGB转灰度
    # 计算深度图梯度
    depth_grad_v, depth_grad_h = image_gradient(depth)

    # 应用掩码筛选有效梯度
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    # 计算基于灰度梯度的权重（边缘敏感加权）
    w_h = torch.exp(-10 * gray_grad_h ** 2)  # 灰度梯度大的区域权重小
    w_v = torch.exp(-10 * gray_grad_v ** 2)

    # 计算加权绝对梯度误差
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
            w_v * torch.abs(depth_grad_v)
    ).mean()

    return err


# tracking阶段总损失计算入口
def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    # 应用曝光补偿参数
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    # 根据配置选择单目/RGBD模式
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


# 单目tracking的RGB损失计算
def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape

    # 创建RGB有效区域掩码（排除暗区）
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask  # 结合预存的梯度掩码

    # 计算加权L1损失
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


# RGBD跟踪的联合损失计算
def get_loss_tracking_rgbd(
        config, image, depth, opacity, viewpoint, initialization=False
):
    # 获取深度权重系数（默认0.95）
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    # 准备深度真值数据
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]  # 添加batch维度

    # 创建深度有效区域掩码
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)  # 不透明度区域

    # 计算RGB损失分量
    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)

    # 计算深度损失分量
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)

    # 返回加权总损失
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


# mapping阶段总损失计算入口
def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    # 处理初始化阶段的特殊情况
    if initialization:
        image_ab = image  # 不应用曝光补偿
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    # 选择单目/RGBD模式
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


# 单目mapping的RGB损失计算
def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape

    # 创建RGB有效区域掩码
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    # 计算普通L1损失
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1_rgb.mean()


# RGBD mapping的联合损失计算
def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    # 获取深度权重系数
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    # 准备真值数据
    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]

    # 创建RGB和深度掩码
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    # 分别计算两个分量
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    # 加权合并损失
    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


# 计算有效深度的中位数统计量
def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()  # 避免修改原张量
    opacity = opacity.detach()

    # 创建有效区域掩码
    valid = depth > 0  # 基本深度有效区域
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)  # 不透明度区域
    if mask is not None:
        valid = torch.logical_and(valid, mask)  # 附加自定义掩码

    # 提取有效深度值
    valid_depth = depth[valid]

    # 根据参数返回统计量
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
