import copy

import torch
import torch.multiprocessing as mp


class FakeQueue:
    """模拟多进程队列的虚拟队列（不实际存储数据）"""

    def put(self, arg):
        """模拟数据放入队列操作（实际丢弃数据）"""
        del arg  # 直接删除参数，不进行任何存储

    def get_nowait(self):
        """模拟非阻塞获取数据（总是抛出空队列异常）"""
        raise mp.queues.Empty  # 抛出标准队列空异常

    def qsize(self):
        """返回虚拟队列大小（始终为0）"""
        return 0

    def empty(self):
        """检查队列是否为空（始终返回True）"""
        return True


def clone_obj(obj):
    """
    深度克隆对象，特别处理Torch张量
    参数:
        obj: 需要克隆的任意对象
    返回:
        深度克隆后的新对象
    """
    # 使用深拷贝创建基础对象副本
    clone_obj = copy.deepcopy(obj)

    # 遍历对象的所有属性
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        # 跳过类属性（property装饰器定义的属性）
        if hasattr(clone_obj.__class__, attr) and isinstance(
                getattr(clone_obj.__class__, attr), property
        ):
            continue

        # 特殊处理Torch张量
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            # 创建脱离计算图的张量副本
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())

    return clone_obj
