export CUDA_VISIBLE_DEVICES=$1
model=$2

HF_ENDPOINT=https://hf-mirror.com python src/dpo.py --model $model > log/$model-DPO.log