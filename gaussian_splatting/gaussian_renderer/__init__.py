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

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh


def render(
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # 如果没有任何高斯点，那就返回None
    if pc.get_xyz.shape[0] == 0:
        return None

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个零张量，后续用于存储屏幕空间上的点，并使其能够计算梯度
    screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
    )
    # 尝试保留梯度信息
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    # 设置光栅化配置，计算水平和垂直视场角的切线值
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 创建光栅化设置对象
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # 创建光栅化器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯模型的3D均值、2D均值、不透明度
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 检查是否提供了预计算的3D协方差，如果没有，将使用光栅化器进行计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 如果预计算颜色不存在，使用SH进行颜色计算
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # 使用SH系数和相机视角计算颜色
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 限制颜色范围
        else:
            shs = pc.get_features  # 如果没有预计算颜色，使用SH特征
    else:
        colors_precomp = override_color  # 如果传入了覆盖颜色，使用它

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 光栅化可见的高斯点到图像，获得它们在屏幕上的半径
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
