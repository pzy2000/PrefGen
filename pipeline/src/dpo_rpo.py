from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import transformers
import torch
from tqdm import tqdm
import re
import os
import argparse
import json
from datasets import load_from_disk, load_dataset
import random, numpy as np
from ollama import chat, ChatResponse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use for code generation.', required=True, type=str)
       
    args = parser.parse_args()
    
    if args.model == "Qwen":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-raw-DPO"
    elif args.model == "Qwen-7B":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-DPO_rpo"
    elif args.model == "DeepSeek-7B":
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/DeepSeek-DPO"
    elif args.model == "Llama":
        model = AutoModelForCausalLM.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438", torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Llama-DPO"
    elif args.model == "Qwen-7B-SFT":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT-DPO_rpo"
    elif args.model == "DeepSeek-7B-SFT":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/DeepSeek-7B-SFT", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/DeepSeek-7B-SFT")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/DeepSeek-7B-SFT-DPO_rpo"
    
    dataset = load_from_disk("/home/pdia/data/GreenCoder/datasets/dpo_solidity_data")


    training_args = DPOConfig(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        learning_rate=5e-6,
        logging_steps=1,
        max_length=2048,
        # eval_strategy="steps",
        save_strategy="epoch",
        rpo_alpha=0.5,
    )


    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["test"],
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)