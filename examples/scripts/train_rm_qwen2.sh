set -x

export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE=/data/nas/guanxinyan/.cache
read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path /data/nas/guanxinyan/OpenRLHF/models/qwen2_7b_rm \
     --ckpt_path /data/nas/guanxinyan/OpenRLHF/models/qwen2_7b_rm \
     --save_steps 200 \
     --logging_steps 1 \
     --eval_steps 200 \
     --train_batch_size 8 \
     --micro_train_batch_size 1 \
     --max_samples 200 \
     --pretrain /cpfs/2926428ee2463e44/user/guanxinyan/fastalign/models/qwen2_7b_ultrachat/checkpoint-1624 \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset Anthropic/hh-rlhf \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
