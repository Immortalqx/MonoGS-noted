import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

import matplotlib

matplotlib.use('Agg')  # 使用无显示设备的后端

from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


# 使用evo工具对轨迹进行评估，计算ATE的RMSE，并生成评估图像
def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    # 使用真实位姿生成3D轨迹对象
    traj_ref = PosePath3D(poses_se3=poses_gt)
    # 使用估计位姿生成3D轨迹对象
    traj_est = PosePath3D(poses_se3=poses_est)
    # 将估计轨迹与真实轨迹对齐，若是单目情况下则校正尺度
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    # 设置位姿关系，提取平移部分用于计算误差
    pose_relation = metrics.PoseRelation.translation_part
    # 将真实轨迹和对齐后的估计轨迹组合成数据元组
    data = (traj_ref, traj_est_aligned)
    # 初始化绝对位置误差（APE）指标，使用平移部分作为误差计算依据
    ape_metric = metrics.APE(pose_relation)
    # 使用数据计算APE指标
    ape_metric.process_data(data)
    # 获取RMSE（均方根误差）统计值
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    # 获取所有统计信息
    ape_stats = ape_metric.get_all_statistics()
    # 记录RMSE ATE值，日志标签为"Eval"
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    # 将统计数据写入JSON文件，文件名中包含标签
    with open(
            os.path.join(plot_dir, "stats_{}.json".format(str(label))),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    # 设置绘图模式为xy平面投影
    plot_mode = evo.tools.plot.PlotMode.xy
    # 创建新的图像
    fig = plt.figure()
    # 根据绘图模式初始化坐标轴
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    # 设置图像标题，显示ATE RMSE值
    ax.set_title(f"ATE RMSE: {ape_stat}")
    # 绘制真实轨迹，使用灰色虚线表示，标签为"gt"
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    # 使用色彩映射绘制对齐后的估计轨迹，颜色代表误差大小，设置最小和最大映射值
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    # 显示图例
    ax.legend()
    # 保存绘制好的图像到指定目录，文件名中包含标签，分辨率为90dpi
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    # 返回计算得到的RMSE ATE值
    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    # 定义一个空字典用于存储轨迹数据
    trj_data = dict()
    # 根据是否为最终评估确定最新帧的索引
    # 如果final为True，则最新帧索引为最后一个关键帧ID加2，否则为加1
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    # 初始化列表用于存储轨迹ID、估计轨迹和真实轨迹
    trj_id, trj_est, trj_gt = [], [], []
    # 初始化列表用于存储估计轨迹和真实轨迹的NumPy数组格式（用于后续评估）
    trj_est_np, trj_gt_np = [], []

    # 定义内部函数，将旋转矩阵R和平移向量T转换为4x4位姿矩阵
    def gen_pose_matrix(R, T):
        # 创建4x4单位矩阵
        pose = np.eye(4)
        # 将单位矩阵的左上角3x3区域赋值为R（转换为NumPy数组）
        pose[0:3, 0:3] = R.cpu().numpy()
        # 将单位矩阵的前三个元素赋值为T（转换为NumPy数组）
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    # 遍历每个关键帧ID
    for kf_id in kf_ids:
        # 从frames中获取当前关键帧对象
        kf = frames[kf_id]
        # 计算估计位姿：先生成位姿矩阵，然后取其逆
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        # 计算真实位姿：先生成真实位姿矩阵，然后取其逆
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        # 将当前帧的唯一ID添加到轨迹ID列表中
        trj_id.append(frames[kf_id].uid)
        # 将估计位姿转换为列表格式并添加到估计轨迹列表中
        trj_est.append(pose_est.tolist())
        # 将真实位姿转换为列表格式并添加到真实轨迹列表中
        trj_gt.append(pose_gt.tolist())

        # 同时保存NumPy数组格式的轨迹，供后续评估使用
        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    # 将轨迹ID、估计轨迹和真实轨迹存入字典中
    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    # 构造保存绘图结果的目录路径，并创建该目录（若不存在）
    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    # 根据是否为最终评估设置标签，若final为True，则标签为"final"，否则格式化为4位数迭代次数
    label_evo = "final" if final else "{:04}".format(iterations)
    # 将轨迹数据写入JSON文件，文件名包含标签
    with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    # 调用evaluate_evo函数对位姿误差进行评估，计算ATE
    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    # 使用wandb记录当前最新帧索引和计算得到的ATE
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    # 返回计算得到的ATE
    return ate


def eval_rendering(
        frames,  # 输入帧数据列表
        gaussians,  # 高斯混合模型数据
        dataset,  # 数据集对象，用于获取真实图像
        save_dir,  # 保存结果的目录
        pipe,  # 渲染管道对象
        background,  # 背景信息
        kf_indices,  # 关键帧索引列表
        iteration="final",  # 迭代标签，默认为"final"
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    # 根据迭代类型确定结束索引，如果是"final"或"before_opt"，则end_idx为最后一帧索引，否则为iteration指定的值
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    # 初始化PSNR、SSIM、LPIPS指标的数组
    psnr_array, ssim_array, lpips_array = [], [], []
    # 初始化LPIPS计算器，采用alex网络并归一化，然后移动到CUDA设备上
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # 遍历从0到end_idx，步长为interval
    for idx in range(0, end_idx, interval):
        # 如果当前索引在关键帧索引中，则跳过此次评估
        if idx in kf_indices:
            continue
        # 保存当前帧的索引
        saved_frame_idx.append(idx)
        # 从frames中获取当前帧
        frame = frames[idx]
        # 从dataset中获取当前帧对应的真实图像及其他数据（后两个变量未使用）
        gt_image, _, _ = dataset[idx]

        # 调用render函数进行渲染，获取渲染结果中的"render"图像
        rendering = render(frame, gaussians, pipe, background)["render"]
        # 将渲染结果限制在0到1之间
        image = torch.clamp(rendering, 0.0, 1.0)

        # 将真实图像从Tensor转换为NumPy数组，调整通道顺序，并转换为8位无符号整数（乘以255）
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        # 将预测图像从Tensor转换为NumPy数组，调整通道顺序，并转换为8位无符号整数
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        # 使用OpenCV将BGR格式转换为RGB格式，并将转换后的图像添加到对应列表中
        img_pred.append(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
        img_gt.append(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))

        # 创建一个掩码，选取真实图像中大于0的像素
        mask = gt_image > 0

        # 计算PSNR，先将预测和真实图像中mask区域的像素提取出来并增加batch维度
        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        # 计算SSIM，同样扩展维度后计算
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        # 计算LPIPS指标
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        # 将各个指标的数值添加到对应的数组中
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        # 如果迭代为"final"或者"before_opt"，则进行图像保存操作
        if iteration == "final" or "before_opt":
            # 构造保存渲染图像的目录路径，并确保该目录存在
            rendering_dir = os.path.join(save_dir, "rendering", iteration)
            os.makedirs(rendering_dir, exist_ok=True)

            # 如果真实图像或预测图像为空，则跳过当前索引
            if gt.size == 0 or pred.size == 0:
                print(f"Skipping idx {idx} due to empty image.")
                continue

            # gt_path = os.path.join(rendering_dir, f"gt_{idx:06d}.png")
            # pred_path = os.path.join(rendering_dir, f"pred_{idx:06d}.png")
            # cv2.imwrite(gt_path, gt)
            # cv2.imwrite(pred_path, pred)

            # 将真实图像和预测图像在水平方向上拼接
            combined_image = np.concatenate((gt, pred), axis=1)
            # 创建一个图像绘制的Figure和坐标轴
            fig, ax = plt.subplots(figsize=(12, 6))
            # 在坐标轴上显示拼接后的图像
            ax.imshow(combined_image)
            # 关闭坐标轴显示
            ax.axis("off")
            # 构造标题信息，包含PSNR、SSIM和LPIPS的数值
            title = (
                f"GT (Left) | Pred (Right)\n"
                f"PSNR: {psnr_score.item():.2f}, "
                f"SSIM: {ssim_score.item():.2f}, "
                f"LPIPS: {lpips_score.item():.2f}"
            )
            # 设置图像标题，字体大小为12，颜色为黑色
            ax.set_title(title, fontsize=12, color="black")

            # 构造拼接图像的保存路径，文件名为6位数字格式
            combined_path = os.path.join(rendering_dir, f"{idx:06d}.png")
            # 保存拼接图像，使用紧凑的边界框和150dpi的分辨率
            plt.savefig(combined_path, bbox_inches="tight", dpi=150)
            # 关闭当前Figure以释放内存
            plt.close(fig)

    # 初始化一个输出字典，用于存储平均指标
    output = dict()
    # 计算并存储平均PSNR
    output["mean_psnr"] = float(np.mean(psnr_array))
    # 计算并存储平均SSIM
    output["mean_ssim"] = float(np.mean(ssim_array))
    # 计算并存储平均LPIPS
    output["mean_lpips"] = float(np.mean(lpips_array))

    # 记录日志，输出平均PSNR、SSIM和LPIPS指标，日志标签为"Eval"
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    # 构造保存PSNR结果的目录路径，并确保该目录存在
    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    # 将输出字典以JSON格式保存到文件中，文件名为"final_result.json"
    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    # 返回输出字典
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
