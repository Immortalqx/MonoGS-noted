import torch
from torch import nn
import cv2

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    # 初始化Camera类，继承自nn.Module
    def __init__(
            self,
            uid,  # 摄像机的唯一标识符
            color,  # 摄像机采集的彩色图像
            depth,  # 摄像机采集的深度图像
            gt_T,  # 摄像机的真实位姿（Ground Truth Transform）
            projection_matrix,  # 摄像机的投影矩阵
            fx,  # 水平方向焦距
            fy,  # 垂直方向焦距
            cx,  # 主点水平方向坐标
            cy,  # 主点垂直方向坐标
            fovx,  # 水平视场角
            fovy,  # 垂直视场角
            image_height,  # 图像高度
            image_width,  # 图像宽度
            device="cuda:0",  # 设备类型，默认为GPU
    ):
        # 调用父类构造函数
        super(Camera, self).__init__()
        # 存储摄像机唯一标识符
        self.uid = uid
        # 存储设备信息
        self.device = device

        # 初始化位姿为单位矩阵
        T = torch.eye(4, device=device)
        # 设置初始旋转矩阵为单位矩阵的前三行前三列
        self.R = T[:3, :3]
        # 设置初始平移向量为单位矩阵的前三行第四列
        self.T = T[:3, 3]
        # 设置真实旋转矩阵，从gt_T中提取
        self.R_gt = gt_T[:3, :3]
        # 设置真实平移向量，从gt_T中提取
        self.T_gt = gt_T[:3, 3]

        # 保存原始彩色图像
        self.original_image = color
        # FIXME 加一个depth的检查，MonoGS的代码似乎默认depth和image的size是相等的
        #  然而对于ScanNet等数据集，depth和image还是有一点差距。。。
        #  这里resize depth真的好吗？？？
        if color.shape[1:] != depth.shape:
            depth = cv2.resize(depth, (color.shape[2], color.shape[1]), interpolation=cv2.INTER_NEAREST)
        # 保存深度图
        self.depth = depth
        # 初始化梯度掩码为空
        self.grad_mask = None

        # 保存内参参数
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # 保存视场角
        self.FoVx = fovx
        self.FoVy = fovy
        # 保存图像尺寸
        self.image_height = image_height
        self.image_width = image_width

        # 定义可训练的摄像机旋转偏移量参数，初始化为0
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        # 定义可训练的摄像机平移偏移量参数，初始化为0
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # 定义曝光参数a，作为可训练参数，初始化为0
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        # 定义曝光参数b，作为可训练参数，初始化为0
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # 将传入的投影矩阵转移到指定设备上并保存
        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        """
        从数据集中初始化Camera对象
        """
        # 从数据集中获取对应索引的彩色图像、深度图和真实位姿
        gt_color, gt_depth, gt_pose = dataset[idx]
        # 返回一个Camera对象，并传入数据集中的内参及其他信息
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        """
        从GUI界面初始化Camera对象
        """
        # 利用getProjectionMatrix2函数生成投影矩阵，并进行转置操作
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        # 返回一个Camera对象，注意此处没有彩色图像和深度图
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        """
        计算世界坐标到视图坐标的变换矩阵，并进行转置
        """
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        """
        计算完整的投影变换矩阵：将世界坐标通过视图变换后，再乘以投影矩阵
        """
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        """
        计算摄像机中心位置
        """
        # 通过世界视图变换矩阵的逆得到摄像机中心，取第4行的前三个元素
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        """
        更新摄像机的旋转矩阵和平移向量
        """
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        """
        根据配置计算图像梯度掩码，用于后续优化或其他用途
        """
        # 从配置中获取边缘阈值
        edge_threshold = config["Training"]["edge_threshold"]

        # 将原始彩色图像转为灰度图（取均值），保留维度
        gray_img = self.original_image.mean(dim=0, keepdim=True)
        # 计算垂直和水平方向的图像梯度
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        # 获取梯度掩码（垂直和水平方向）
        mask_v, mask_h = image_gradient_mask(gray_img)
        # 将梯度乘以对应的掩码
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        # 计算梯度强度（梯度的模长）
        img_grad_intensity = torch.sqrt(gray_grad_v ** 2 + gray_grad_h ** 2)

        # 如果数据集类型为"replica"，进行分块处理
        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32  # 定义分块数量
            multiplier = edge_threshold  # 阈值乘子
            _, h, w = self.original_image.shape  # 获取图像高度和宽度
            # 遍历每个块
            for r in range(row):
                for c in range(col):
                    # 提取当前块的梯度强度
                    block = img_grad_intensity[
                            :,
                            r * int(h / row): (r + 1) * int(h / row),
                            c * int(w / col): (c + 1) * int(w / col),
                            ]
                    # 计算当前块梯度的中值
                    th_median = block.median()
                    # 将高于中值乘以乘子的部分置为1
                    block[block > (th_median * multiplier)] = 1
                    # 将低于等于中值乘以乘子的部分置为0
                    block[block <= (th_median * multiplier)] = 0
            # 将计算后的梯度强度赋值给grad_mask
            self.grad_mask = img_grad_intensity
        else:
            # 对于其他数据集，计算整体图像梯度的中值
            median_img_grad_intensity = img_grad_intensity.median()
            # 根据中值和边缘阈值生成掩码，梯度大于阈值的部分为True
            self.grad_mask = (
                    img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        """
        清理Camera对象中的部分数据，释放内存
        """
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
