set -x 
export PATH=$HOME/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 deepspeed ../inference.py \
    --pretrain "$HOME/scratch/data/llama_hf/7B" \
    --bf16 \
    --load_model "./ckpt/7b_llama_1024/sft_model.pt"\
    --input "$1"