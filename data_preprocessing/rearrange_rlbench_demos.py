# 此文件用于重新组织RLBench演示数据的目录结构
# 主要功能:
# 1. 将所有变体(variations)下的演示数据按照变体ID重新组织
# 2. 为每个变体创建独立的目录结构
# 3. 复制和重命名相关的描述文件和演示数据

import os
from subprocess import call
import pickle
from pathlib import Path
import shutil
from tqdm import tqdm

import tap


class Arguments(tap.Tap):
    # 定义命令行参数类,接收根目录路径
    root_dir: Path


def main(root_dir, task):
    print(f"\nProcessing task: {task}")
    # 获取指定任务下所有变体的episodes目录列表
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    # 用字典记录每个变体已处理的演示数量
    seen_variations = {}
    
    # 遍历所有变体目录
    for variation in tqdm(variations, desc=f'Processing {task}'):
        # 从文件名获取演示编号
        num = int(variation.replace('episode', ''))
        # 加载变体编号信息
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        # 创建对应变体的目录结构
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        # 记录该变体下的演示编号
        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        # 检查并复制变体描述文件
        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            # 如果目标目录已存在描述文件,验证内容一致性
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            # 如果是符号链接则删除
            if os.path.islink(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
                os.unlink(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl')
            # 复制描述文件到新位置
            shutil.copyfile(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl')

        # 计算新的演示ID并设置源目录和目标目录
        ep_id = len(seen_variations[variation]) - 1
        src_dir = f'{root_dir}/{task}/all_variations/episodes/episode{num}'
        dst_dir = f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'
        
        # 如果目标目录存在,则删除(包括符号链接)
        if os.path.exists(dst_dir):
            if os.path.islink(dst_dir):
                os.unlink(dst_dir)
            else:
                shutil.rmtree(dst_dir)
        
        # 复制整个演示目录到新位置
        shutil.copytree(src_dir, dst_dir)
        print(f'Copied: {src_dir} -> {dst_dir}')


if __name__ == '__main__':
    # 解析命令行参数
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    print(f"Starting processing with root directory: {root_dir}")
    # 获取所有任务目录(排除zip文件)
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    print(f"Found {len(tasks)} tasks to process")
    # 遍历处理每个任务
    for task in tqdm(tasks, desc='Total progress'):
        main(root_dir, task)
