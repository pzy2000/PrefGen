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


class CustomDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ):
        alpha = 1.0
        beta = 1.0
        lambda_coef = 0.1
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        gas_c = inputs.get("gas_chosen"); gas_r = inputs.get("gas_rejected")
        vul_c = inputs.get("vul_chosen"); vul_r = inputs.get("vul_rejected")
        gas_c = torch.tensor(gas_c, dtype=torch.float, device=base_loss.device) if gas_c is not None else None
        gas_r = torch.tensor(gas_r, dtype=torch.float, device=base_loss.device) if gas_r is not None else None
        vul_c = torch.tensor(vul_c, dtype=torch.float, device=base_loss.device) if vul_c is not None else None
        vul_r = torch.tensor(vul_r, dtype=torch.float, device=base_loss.device) if vul_r is not None else None
        extra_loss = 0.0
        if gas_c is not None and gas_r is not None:
            R_g = -(gas_c - gas_r)
            extra_loss += alpha * (-R_g.mean())
        if vul_c is not None and vul_r is not None:
            safe_c = 1.0 - vul_c
            safe_r = 1.0 - vul_r
            R_v = safe_c - safe_r
            extra_loss += beta * (-R_v.mean())

        total_loss = base_loss + lambda_coef * extra_loss
        return (total_loss, outputs) if return_outputs else total_loss




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use for code generation.', required=True, type=str)
       
    args = parser.parse_args()
    
    if args.model == "Qwen":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-pzyo"
    elif args.model == "Qwen-7B":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-pzyo"
    elif args.model == "DeepSeek-7B":
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/DeepSeek-7B-DPO"
    elif args.model == "DeepSeek-7B-SFT":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/DeepSeek-SFT", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/DeepSeek-SFT")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/DeepSeek-7B-SFT-DPO"
    elif args.model == "Llama":
        model = AutoModelForCausalLM.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438", torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Llama-DPO"
    elif args.model == "Qwen-7B-SFT":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT-pzyo"
    elif args.model == "Qwen-7B-DPO_rpo":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-DPO_rpo", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-DPO_rpo")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-DPO_rpo-pzyo"
        
    
    dataset = load_from_disk("/home/pdia/data/GreenCoder/datasets/pzyo_solidity_data")


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
        rpo_alpha=0.5,
        # eval_strategy="steps",
        save_strategy="epoch",
        # eval_steps=500,
        # save_steps=1000,
    )


    # trainer = DPOTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     # eval_dataset=dataset["test"],
    #     processing_class=tokenizer
    # )
    # exit()
    trainer = CustomDPOTrainer(model=model, args=training_args, 
                            train_dataset=dataset["train"], processing_class=tokenizer)
    trainer.train()
    trainer.save_model(training_args.output_dir)