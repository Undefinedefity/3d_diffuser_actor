# 导入所需的Python标准库
import random  # 用于生成随机数
import itertools  # 用于生成迭代器
from typing import Tuple, Dict, List  # 用于类型注解
import pickle  # 用于序列化和反序列化Python对象
from pathlib import Path  # 用于处理文件路径
import json  # 用于处理JSON数据

# 导入第三方库
import blosc  # 用于数据压缩
from tqdm import tqdm  # 用于显示进度条
import tap  # 用于命令行参数解析
import torch  # PyTorch深度学习框架
import numpy as np  # 用于数值计算
import einops  # 用于张量操作
from rlbench.demo import Demo  # RLBench仿真环境的Demo类

# 导入自定义工具函数
from utils.utils_with_rlbench import (
    RLBenchEnv,  # RLBench环境封装
    keypoint_discovery,  # 关键帧发现
    obs_to_attn,  # 观察转注意力
    transform,  # 数据转换
)


# 定义命令行参数类
class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "c2farm"  # 数据目录
    seed: int = 2  # 随机种子
    tasks: Tuple[str, ...] = ("stack_wine",)  # 任务名称
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist", "front")  # 相机视角
    image_size: str = "256,256"  # 图像大小
    output: Path = Path(__file__).parent / "datasets"  # 输出目录
    max_variations: int = 199  # 最大变体数量
    offset: int = 0  # 偏移量
    num_workers: int = 0  # 工作进程数
    store_intermediate_actions: int = 1  # 是否存储中间动作


# 从演示中获取注意力索引
def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...]
) -> List[Dict[str, Tuple[int, int]]]:
    frames = keypoint_discovery(demo)  # 发现关键帧

    frames.insert(0, 0)  # 在开头插入初始帧
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]  # 返回每个相机的注意力索引


# 获取观察数据
def get_observation(task_str: str, variation: int,
                    episode: int, env: RLBenchEnv,
                    store_intermediate_actions: bool):
    demos = env.get_demo(task_str, variation, episode)  # 获取演示
    demo = demos[0]  # 取第一个演示

    key_frame = keypoint_discovery(demo)  # 发现关键帧
    key_frame.insert(0, 0)  # 在开头插入初始帧

    keyframe_state_ls = []  # 关键帧状态列表
    keyframe_action_ls = []  # 关键帧动作列表
    intermediate_action_ls = []  # 中间动作列表

    # 遍历所有关键帧
    for i in range(len(key_frame)):
        state, action = env.get_obs_action(demo._observations[key_frame[i]])  # 获取状态和动作
        state = transform(state)  # 转换状态
        keyframe_state_ls.append(state.unsqueeze(0))  # 添加状态
        keyframe_action_ls.append(action.unsqueeze(0))  # 添加动作

        # 如果需要存储中间动作且不是最后一帧
        if store_intermediate_actions and i < len(key_frame) - 1:
            intermediate_actions = []  # 中间动作列表
            # 获取两个关键帧之间的所有动作
            for j in range(key_frame[i], key_frame[i + 1] + 1):
                _, action = env.get_obs_action(demo._observations[j])
                intermediate_actions.append(action.unsqueeze(0))
            intermediate_action_ls.append(torch.cat(intermediate_actions))

    return demo, keyframe_state_ls, keyframe_action_ls, intermediate_action_ls


# 定义数据集类
class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: Arguments):
        # 初始化RLBench环境
        self.env = RLBenchEnv(
            data_path=args.data_dir,  # 数据路径
            image_size=[int(x) for x in args.image_size.split(",")],  # 图像大小
            apply_rgb=True,  # 使用RGB图像
            apply_pc=True,  # 使用点云
            apply_cameras=args.cameras,  # 使用的相机
        )

        tasks = args.tasks  # 任务列表
        variations = range(args.offset, args.max_variations)  # 变体范围
        self.items = []  # 数据项列表
        # 遍历所有任务和变体组合
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"  # 获取episode目录
            episodes = [
                (task_str, variation, int(ep.stem[7:]))  # 提取episode编号
                for ep in episodes_dir.glob("episode*")  # 获取所有episode文件
            ]
            self.items += episodes  # 添加到数据项列表

        self.num_items = len(self.items)  # 数据项总数

    def __len__(self) -> int:
        return self.num_items  # 返回数据集长度

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]  # 获取任务、变体和episode
        taskvar_dir = args.output / f"{task}+{variation}"  # 创建输出目录
        taskvar_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        # 获取观察数据
        (demo,
         keyframe_state_ls,
         keyframe_action_ls,
         intermediate_action_ls) = get_observation(
            task, variation, episode, self.env,
            bool(args.store_intermediate_actions)
        )

        # 重排状态数据的维度
        state_ls = einops.rearrange(
            keyframe_state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))  # 帧ID列表
        num_frames = len(frame_ids)  # 帧数量
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras)  # 获取注意力索引

        # 创建状态字典
        state_dict: List = [[] for _ in range(6)]
        print("Demo {}".format(episode))  # 打印当前处理的Demo
        state_dict[0].extend(frame_ids)  # 添加帧ID
        state_dict[1] = state_ls[:-1].numpy()  # 添加状态数据
        state_dict[2].extend(keyframe_action_ls[1:])  # 添加关键帧动作
        state_dict[3].extend(attn_indices)  # 添加注意力索引
        state_dict[4].extend(keyframe_action_ls[:-1])  # 添加机械臂位置
        state_dict[5].extend(intermediate_action_ls)   # 添加中间轨迹动作

        # 保存数据
        with open(taskvar_dir / f"ep{episode}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))  # 压缩并保存数据


# 主程序入口
if __name__ == "__main__":
    args = Arguments().parse_args()  # 解析命令行参数

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = Dataset(args)  # 创建数据集
    dataloader = torch.utils.data.DataLoader(  # 创建数据加载器
        dataset,
        batch_size=1,  # 批次大小
        num_workers=args.num_workers,  # 工作进程数
        collate_fn=lambda x: x,  # 数据整理函数
    )

    for _ in tqdm(dataloader):  # 遍历数据集
        continue
