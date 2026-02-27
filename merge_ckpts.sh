#!/bin/bash

# source /data/qingnan/miniconda3/etc/profile.d/conda.sh

# conda activate verl_train_sa_1

# cd /verl

ckpt_base_dir="/data/qingnan/temp/verl_grpo_example_CW_2/qwen3_8b_deepseek_final_two_stage_grm_one_20260206_20251224_170159"

for step in $(seq 220 10 260); do
    /root/miniconda3/envs/verl/bin/python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${ckpt_base_dir}/global_step_${step}/actor \
        --target_dir ${ckpt_base_dir}/global_step_${step}/actor/huggingface
done

/root/miniconda3/envs/verl/bin/python3 scripts/legacy_model_merger.py merge --backend fsdp --local_dir /data/qingnan/adora_geo/global_step_60/actor --target_dir /data/qingnan/adora_geo/global_step_60/actor/huggingface