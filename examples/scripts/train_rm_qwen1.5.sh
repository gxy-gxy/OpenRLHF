set -x

export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE=/data/nas/guanxinyan/.cache
read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path /data/nas/guanxinyan/OpenRLHF/models/qwen1.4_14b_rm \
     --ckpt_path /data/nas/guanxinyan/OpenRLHF/models/qwen1.4_14b_rm \
     --save_steps 1000 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --train_batch_size 32 \
     --micro_train_batch_size 4 \
     --pretrain /cpfs/2926428ee2463e44/user/guanxinyan/fastalign/models/qwen1.5_14b_ift_eft_filter_alpaca/checkpoint-93 \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,lmsys/chatbot_arena_conversations \
     --dataset_probs 0.72,0.12,0.16 \
     --flash_attn \
     --gradient_checkpointing \
     --load_checkpoint
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
