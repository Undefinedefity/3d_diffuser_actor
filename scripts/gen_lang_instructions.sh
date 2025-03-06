#!/bin/bash

# 添加PYTHONPATH环境变量，确保Python能找到项目根目录
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python data_preprocessing/preprocess_calvin_instructions.py \
  --output instructions/calvin_task_ABC_D/validation.pkl \
  --model_max_length 16 \
  --annotation_path ./calvin/dataset/task_ABC_D/validation/lang_annotations/auto_lang_ann.npy

python data_preprocessing/preprocess_calvin_instructions.py \
  --output instructions/calvin_task_ABC_D/training.pkl \
  --model_max_length 16 \
  --annotation_path ./calvin/dataset/task_ABC_D/training/lang_annotations/auto_lang_ann.npy

python data_preprocessing/preprocess_rlbench_instructions.py \
  --tasks place_cups close_jar insert_onto_square_peg light_bulb_in \
  meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location \
  push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe \
  reach_and_drag slide_block_to_color_target stack_blocks stack_cups \
  sweep_to_dustpan_of_size turn_tap \
  --output instructions.pkl
