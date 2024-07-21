set -x
# rm -r /data/nas/guanxinyan/OpenRLHF/models/qwen2_7b_rm

export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE=/data/nas/guanxinyan/.cache

read -r -d '' training_commands <<EOF
../train_value_model.py \
     --save_path ${OUTPUT_DIR:-"/data/nas/guanxinyan/OpenRLHF/models/qwen2_7b_rm"} \
     --ckpt_path ${OUTPUT_DIR:-"/data/nas/guanxinyan/OpenRLHF/models/qwen2_7b_rm"} \
     --save_steps ${SAVE_STEPS:-1000} \
     --logging_steps 1 \
     --eval_steps ${EVAL_STEPS:-200} \
     --train_batch_size ${TRAIN_BATCH_SIZE:-16} \
     --micro_train_batch_size 1 \
     --response_micro_batch_size ${RESPONSE_MICRO_BATCH_SIZE:-1} \
     --input_key prompt \
     --output_key response \
     --apply_chat_template \
     --pretrain ${PRETRAIN:-"/cpfs/2926428ee2463e44/user/guanxinyan/hf_models/Qwen2-1.5B"} \
     --bf16 \
     --max_epochs ${EPOCHS:-1} \
     --max_len ${MAX_LEN:-2048} \
     --zero_stage 3 \
     --learning_rate 5e-6 \
     --dataset ${DATASET:-"/data/nas/guanxinyan/OpenRLHF/data/toy.jsonl"} \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
