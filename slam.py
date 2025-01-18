import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        # ################## SYSTEM STEP 2.1 : 初始化MonoGS ################## START

        # 初始化start 和 end，用于记录 GPU 上某个时间点的时间戳
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # 记录开始的时间
        start.record()

        # ################## SYSTEM STEP 2.1.1 : 加载参数 ################## START

        self.config = config
        self.save_dir = save_dir
        # 将字典转换为Munch对象，之后可以使用点操作"."符访问字典中的元素
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )  # self.model_params实际上没有用，后面还是用的model_params

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]  # 是否使用求谐函数
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # ################## SYSTEM STEP 2.1.1 : 加载参数 ################## END

        # ################## SYSTEM STEP 2.1.2 : 初始化 ################## START

        # 初始化3DGS
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        # 给3DGS设置初始的学习率
        self.gaussians.init_lr(6.0)
        # 初始化dataloader
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        # 初始化3DGS训练的各种参数
        self.gaussians.training_setup(opt_params)
        # 设置背景的颜色
        bg_color = [0, 0, 0]  # 默认为黑色的背景
        # bg_color = [1, 1, 1] # 白色背景，注意这里不是255
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 下面创建了一些队列，主要用于不同进程之间的通信或数据交换
        # mp.Queue() 提供了一个多进程安全的队列实现，可以被多个进程同时访问，而不会导致数据不一致或冲突
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # 初始化SLAM的前端
        self.frontend = FrontEnd(self.config)
        # 初始化SLAM的后端
        self.backend = BackEnd(self.config)

        # 指定前端的一系列参数
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # 指定后端的一系列参数
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.set_hyperparams()

        # 指定UI界面的一系列参数
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        # ################## SYSTEM STEP 2.1.2 : 初始化 ################## END

        # ################## SYSTEM STEP 2.1.3 : 启动各个进程 ################## START

        # 为什么先创建 backend_process, 再创建 gui_process？
        # 改了一下顺序跑代码，看起来是没有影响的。
        # backend_process = mp.Process(target=self.backend.run) # 官方的顺序
        if self.use_gui:
            # 创建一个 GUI 进程，目标函数为 slam_gui.run，传递参数 self.params_gui
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            # 启动 GUI 进程
            gui_process.start()
            # 等待5秒，主要是等 GUI 界面加载好
            time.sleep(5)

        # 创建一个多进程对象 backend_process，目标函数为 self.backend.run
        backend_process = mp.Process(target=self.backend.run)  # 我改动的顺序
        # 启动 backend_process 进程
        backend_process.start()

        # 主进程运行frontend
        self.frontend.run()

        # 前端运行结束了，利用队列传递信息，让后端暂停
        backend_queue.put(["pause"])

        # 记录结束的时间
        end.record()

        # 在CUDA设备上同步所有流（streams）
        torch.cuda.synchronize()
        # empty the frontend queue

        # ################## SYSTEM STEP 2.1.3 : 启动各个进程 ################## END

        # ################## SYSTEM STEP 2.1.4 : 评估结果 ################## START

        # 下面是官方实现
        # N_frames = len(self.frontend.cameras)
        # FPS = N_frames / (start.elapsed_time(end) * 0.001)
        # Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        # Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        # 下面是被调整过的实现
        TIME = start.elapsed_time(end) * 0.001  # 计算总耗时
        FPS = len(self.frontend.cameras) / (start.elapsed_time(end) * 0.001)  # 计算平均FPS
        Log("Total time", TIME, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        # 评估渲染结果
        # TODO 这部分留到后面再注释！
        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        # 为什么要放在这个地方join呢？为什么不放在前面呢？
        # 如果需要评估渲染结果的话，还需要后端提供一些数据，不能够放到前面就join了。
        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

        # ################## SYSTEM STEP 2.1.4 : 评估结果 ################## END

    def run(self):
        pass


if __name__ == "__main__":
    # ################## SYSTEM STEP 1 : 参数加载、信息打印 ################## START

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    # sys.argv[1:] 表示从第一个元素（脚本名称之后）到列表末尾的所有元素，即命令行参数的列表
    # 对于python slam.py --config configs/mono/tum/fr1_desk.yaml
    # sys.argv      ['slam.py', '--config', 'configs/mono/tum/fr1_desk.yaml']
    # sys.argv[1:]  ['--config', 'configs/mono/tum/fr1_desk.yaml']
    args = parser.parse_args(sys.argv[1:])

    # 设置多进程的启动方法
    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        if config["Dataset"]["type"] == "replica":
            save_dir = os.path.join(
                config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
            )
        else:
            save_dir = os.path.join(
                config["Results"]["save_dir"], path[-2] + "_" + path[-1], current_datetime
            )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    # ################## SYSTEM STEP 1 : 参数加载、信息打印 ################## END

    # ################## SYSTEM STEP 2 : 运行MonoGS ################## START

    slam = SLAM(config, save_dir=save_dir)

    slam.run()  # 空函数
    wandb.finish()

    # All done
    Log("Done.")

    # ################## SYSTEM STEP 2 : 运行MonoGS ################## END
