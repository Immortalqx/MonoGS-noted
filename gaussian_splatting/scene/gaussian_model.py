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

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        # 初始化球谐次数和最大球谐次数
        self.active_sh_degree = 0  # 当前激活的球谐次数，初始为0
        self.max_sh_degree = sh_degree  # 允许的最大球谐次数

        # 初始化3D高斯模型的各项参数
        self._xyz = torch.empty(0, device="cuda")  # 3D高斯的中心位置（均值）
        self._features_dc = torch.empty(0, device="cuda")  # 第一个球谐系数，用于表示基础颜色
        self._features_rest = torch.empty(0, device="cuda")  # 其余的球谐系数，用于表示颜色的细节和变化
        self._scaling = torch.empty(0, device="cuda")  # 3D高斯的尺度参数，控制高斯的宽度
        self._rotation = torch.empty(0, device="cuda")  # 3D高斯的旋转参数，用四元数表示
        self._opacity = torch.empty(0, device="cuda")  # 3D高斯的不透明度，控制可见性
        self.max_radii2D = torch.empty(0, device="cuda")  # 在2D投影中，每个高斯的最大半径
        self.xyz_gradient_accum = torch.empty(0, device="cuda")  # 用于累积3D高斯中心位置的梯度

        # TODO 待确定的两个参数
        self.unique_kfIDs = torch.empty(0).int()  # 这个看起来是keyframe的id，但为什么用torch？
        self.n_obs = torch.empty(0).int()  # 这个是什么？每一帧被观察的次数？

        self.optimizer = None  # 优化器，用于调整上述参数以改进模型

        # 对应原版3DGS的def setup_functions(self)
        # 初始化一些激活函数，这些激活函数被包装一层get_scaling()、get_rotation()、get_covariance()等，在render渲染的时候被调用
        # 用指数函数确保尺度参数非负
        self.scaling_activation = torch.exp
        # 尺度参数的逆激活函数，对数函数
        self.scaling_inverse_activation = torch.log

        # 协方差矩阵的激活函数
        self.covariance_activation = self.build_covariance_from_scaling_rotation

        # 用sigmoid函数确保不透明度在0到1之间
        self.opacity_activation = torch.sigmoid
        # 不透明度的逆激活函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 用于标准化旋转参数的函数
        self.rotation_activation = torch.nn.functional.normalize

        # 与原版3DGS相比新增内容
        self.config = config
        self.ply_input = None

        # FIXME 这玩意怎么只在高斯初始化用一下？能不能扩展到全局的isotropic？
        self.isotropic = False

    # FIXME 下面可以是static，这里MonoGS稍微调整了原始3DGS的代码顺序
    # 定义构建3D高斯协方差矩阵的函数
    def build_covariance_from_scaling_rotation(
            self, scaling, scaling_modifier, rotation
    ):
        # 计算实际的协方差矩阵
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        # 提取对称部分
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 从单张图像创建点云，如果depthmap为None且为mono模式，会随机初始化一个深度图
    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                                    np.ones_like(depth_raw)
                                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                                    * 0.05
                            ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    # 从有depth的图像创建高斯点云
    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]

        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        # 从颜色图像和深度图像创建 RGBD 图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        # 获取世界坐标到摄像机坐标的转换矩阵，并转换为 NumPy 数组
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()

        # 从 RGBD 图像和相机参数创建点云
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        # 随机降采样点云
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        # 获取降采样后的点云坐标和颜色
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        # 创建input_ply对象并存储在 `ply_input` 属性中，类似gaussian splatting中保存的input.ply
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        # 将点云坐标转换为 Torch 张量并移到 GPU
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

        # 将点云颜色转换为球谐函数表示并移到 GPU
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        # 初始化颜色特征张量并将其移到 GPU
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        # 将球谐函数颜色赋值给特征张量
        features[:, :3, 0] = fused_color
        # 将其他部分初始化为 0
        features[:, 3:, 1:] = 0.0

        # 计算每个点的距离平方，并进行裁剪以避免小值，然后调整点大小
        dist2 = (
                torch.clamp_min(
                    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                    0.0000001,
                )
                * point_size
        )
        # 计算尺度并取对数平方根
        scales = torch.log(torch.sqrt(dist2))[..., None]
        # 如果不是各向同性，则扩展尺度到 3 维
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        # 初始化旋转张量为全零，并将第一个元素设为 1（表示单位四元数）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 计算每个点的不透明度并取逆 sigmoid
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    # 新增高斯点云，类似原版3DGS的densify_and_clone + create_from_pcd
    def extend_from_pcd(
            self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    # 从RGB、RGB-D序列来扩充高斯点云
    def extend_from_pcd_seq(
            self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        # FIXME 在函数中定义了新的变量！原版3DGS是有在__init__()中定义的。
        # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        self.percent_dense = training_args.percent_dense
        # 初始化用于累积3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 这个变量应该是用来记录每个高斯投影到了多少帧的
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        # 创建优化器，这里使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 创建学习率调度器，用于对中心点位置的学习率进行调整
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        # TODO 又在其他函数定义新的类变量。
        #  为什么不按照原版3DGS的来，而是使用help函数？
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """
        根据当前的迭代次数(iteration)动态调整xyz参数的学习率
        Learning rate scheduling per step
        """
        # 遍历优化器中的所有参数组
        for param_group in self.optimizer.param_groups:
            # 找到名为"xyz"的参数组，即3D高斯分布中心位置的参数
            if param_group["name"] == "xyz":
                # 使用xyz_scheduler_args函数（一个根据迭代次数返回学习率的调度函数）计算当前迭代次数的学习率
                # lr = self.xyz_scheduler_args(iteration)

                # TODO 上面是原版3DGS的做法，下面是MonoGS的做法，WHY？
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                # 将计算得到的学习率应用到xyz参数组
                param_group["lr"] = lr
                return lr

    # 保存点云用的
    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        # TODO 这里为什么不按原版3DGS的来呢
        #  原版似乎是倾向于放弃opacity已经很低的高斯，而MonoGS想抢救一下，让大家都重新来一遍。。。
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
            self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        # TODO 不理解为什么这里要给non-visible的高斯设置一个比较大的opacity
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将指定的参数张量替换到优化器中，这主要用于更新模型的某些参数（例如不透明度）并确保优化器使用新的参数值。
        :param tensor: 新的参数张量。
        :param name: 参数的名称，用于在优化器的参数组中定位该参数。
        :return: 包含已更新参数的字典。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        删除不符合要求的3D高斯分布在优化器中对应的参数
        :param mask: 一个布尔张量，表示需要保留的3D高斯分布。
        :return: 更新后的可优化张量字典。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # 更新优化器状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧状态并更新参数
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        删除不符合要求的3D高斯分布。
        :param mask: 一个布尔张量,表示需要删除的3D高斯分布。
        """
        # 生成有效点的掩码并更新优化器中的参数，调用_prune_optimizer()
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新各参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新累积梯度和其他相关张量（这里比原版3DGS多了几个参数）
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的参数张量添加到优化器的参数组中
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
            self,
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=None,
            new_n_obs=None,
    ):
        """
        将新生成的3D高斯分布的属性添加到模型的参数中/一个字典中
        """
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重新初始化一些用于累计梯度、最大2D半径等的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # 简单的拼接剩下的变量（这个变量应该是放在CPU上？）
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        对那些梯度超过一定阈值且尺度大于一定阈值的3D高斯进行分割操作。
        这意味着这些高斯可能过于庞大，覆盖了过多的空间区域，需要分割成更小的部分以提升细节。

        这是原版3DGS的策略，MonoGS在这里仅仅同步了unique_kfIDs、n_obs，没有其他的修改
        """
        # 获取初始点的数量
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        # 选择满足条件的点
        # 创建一个掩码，标记那些梯度大于等于指定阈值的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 一步过滤掉那些缩放（scaling）大于一定百分比的场景范围的点
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # 计算新高斯分布的属性，其中 stds 是点的缩放，means 是均值
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        # 使用均值和标准差生成样本
        samples = torch.normal(mean=means, std=stds)
        # 为每个点构建旋转矩阵，并将其重复 N 次
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        # 计算新的位置，将旋转后的样本点添加到原始点的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        # 调整尺度并保持其他属性
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        # 关键帧和观察次数也是和原来一样
        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        # 将分割得到的新高斯分布的属性添加到场景中
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        # 根据修剪过滤器，删除原有过大的高斯分布
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对那些梯度超过一定阈值且尺度小于一定阈值的3D高斯进行克隆操作。
        这意味着这些高斯在空间中可能表示的细节不足，需要通过克隆来增加细节。

        这是原版3DGS的策略，MonoGS在这里仅仅同步了unique_kfIDs、n_obs，没有其他的修改
        """
        # 选择满足条件的点
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）
        # 的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # MonoGS新增的两行
        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]

        # 将克隆得到的新的密集化点的相关特征保存在一个字典中
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        对3D高斯分布进行密集化和修剪的操作
        :param max_grad: 梯度的最大阈值，用于判断是否需要克隆或分割。
        :param min_opacity: 不透明度的最小阈值,低于此值的3D高斯将被删除。
        :param extent: 场景的尺寸范围，用于评估高斯分布的大小是否合适。
        :param max_screen_size: 最大屏幕尺寸阈值，用于修剪过大的高斯分布。
        """

        # 计算3D高斯中心的累积梯度并修正NaN值
        grads = self.xyz_gradient_accum / self.denom  # 计算密度估计的梯度
        grads[grads.isnan()] = 0.0  # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性

        # 根据梯度和尺寸阈值进行克隆或分割操作
        # ************ 自适应密度控制的两重要部分：densify_and_clone和densify_and_split ************
        self.densify_and_clone(grads, max_grad, extent)  # 对under reconstruction的区域进行稠密化和复制操作
        self.densify_and_split(grads, max_grad, extent)  # 对over reconstruction的区域进行稠密化和分割操作

        # 创建修剪掩码以删除不必要的3D高斯分布
        # 创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # 如果设置了相机的范围
        if max_screen_size:
            # 创建一个掩码，标记在图像空间中半径大于指定阈值的点
            big_points_vs = self.max_radii2D > max_screen_size
            # 创建一个掩码，标记在世界空间中尺寸大于指定阈值的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        # 根据掩码mask,删减points
        self.prune_points(prune_mask)
        # TODO 原版都在这里清理cuda缓存，MonoGS这里为什么把代码给删了？？？
        #         torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
