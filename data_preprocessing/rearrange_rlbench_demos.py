import os
from subprocess import call
import pickle
from pathlib import Path
import shutil
from tqdm import tqdm

import tap


class Arguments(tap.Tap):
    root_dir: Path


def main(root_dir, task):
    print(f"\nProcessing task: {task}")
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    seen_variations = {}
    for variation in tqdm(variations, desc=f'Processing {task}'):
        num = int(variation.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            if os.path.islink(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
                os.unlink(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl')
            shutil.copyfile(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl')

        ep_id = len(seen_variations[variation]) - 1
        src_dir = f'{root_dir}/{task}/all_variations/episodes/episode{num}'
        dst_dir = f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'
        
        if os.path.exists(dst_dir):
            if os.path.islink(dst_dir):
                os.unlink(dst_dir)
            else:
                shutil.rmtree(dst_dir)
        
        shutil.copytree(src_dir, dst_dir)
        print(f'Copied: {src_dir} -> {dst_dir}')


if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    print(f"Starting processing with root directory: {root_dir}")
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    print(f"Found {len(tasks)} tasks to process")
    for task in tqdm(tasks, desc='Total progress'):
        main(root_dir, task)
