#!/usr/bin/env sh

train_data_path='./configs/data.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=/your/path/to/checkpoints
global_batch_size=8
micro_batch_size=1
snr_type=lognorm
lr=1
wd=0.01
precision=bf16
training_type=full_model

dir_name=lumina_results
mkdir -p "$dir_name"

python -u finetune_accessory.py \
    --master_port 18187 \
    --global_bsz_1024 ${global_batch_size} \
    --micro_bsz_1024 ${micro_batch_size} \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 --wd ${wd} \
    --data_path ${train_data_path} \
    --results_dir "$dir_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 5000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20230122 \
    --num_workers 12 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    --training_type ${training_type}
