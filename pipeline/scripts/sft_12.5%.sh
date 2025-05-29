export CUDA_VISIBLE_DEVICES=$1
model=$2
HF_ENDPOINT=https://hf-mirror.com python src/sft_12.5%.py --model $model > log/$model-SFT.log