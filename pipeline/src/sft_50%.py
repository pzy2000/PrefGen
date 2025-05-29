# from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import argparse
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk
from torch import tensor


class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if num_items_in_batch is not None:
            num_items_in_batch_tensor = tensor(num_items_in_batch, dtype=loss.dtype, device=loss.device)
            loss = loss / num_items_in_batch_tensor

        return (loss, outputs) if return_outputs else loss
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use for code generation.', required=True, type=str)
       
    args = parser.parse_args()
    # device_map = {"": 0}  # Puts all layers on cuda:0
    # model = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-Coder-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map=device_map
    # )

    if args.model == "Qwen":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                                    #   attn_implementation="flash_attention_2",
                                                        torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", padding_side="left")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-SFT"
        response_template = "<|im_start|>assistant"
    elif args.model == "Qwen-7B":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", 
                                                    #  attn_implementation="flash_attention_2",
                                                     torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", padding_side="left")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT_50%"
        response_template = "<|im_start|>assistant"
    elif args.model == "Qwen-7B-SFT":
        model = AutoModelForCausalLM.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT", 
                                                    #  attn_implementation="flash_attention_2",
                                                     torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT", padding_side="left")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Qwen-7B-SFT-Round2_50%"
        response_template = "<|im_start|>assistant"
    elif args.model == "Llama":
        model = AutoModelForCausalLM.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("/home/xinyin/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/9b58d4a36161a1e49ecf0a69d20b2736fef8e438")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/Llama-SFT"
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    elif args.model == "DeepSeek-7B":
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", padding_side="left")
        output_dir = "/home/pdia/data/GreenCoder/saved_models/DeepSeek-SFT"
        response_template = "<|begin▁of▁sentence|>assistant"


        
    dataset = load_from_disk("/home/pdia/data/GreenCoder/datasets/sft_solidity_data_50%")
    
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        logging_steps=1,
        max_length=2048,
        eval_strategy="epoch",
        save_strategy="epoch",
        # learning_rate=5e-6,
        # lr_scheduler_type="cosine",
        load_best_model_at_end=True,
    )

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        # train_dataset=dataset,
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)