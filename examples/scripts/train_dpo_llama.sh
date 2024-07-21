set -x 

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo\
     --save_path ./ckpt/qwen1.5_7b_test \
     --save_steps 100 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 16 \
     --micro_train_batch_size 1 \
     --pretrain /cpfs/2926428ee2463e44/user/guanxinyan/hf_models/Qwen1.5-7B \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset /data/nas/guanxinyan/OpenRLHF/data/ultrafeedback.json \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload \
     --prompt_key prompt \
     --chosen_key chosen \
     --rejected_key rejected
EOF
    # --wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload 
    # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
