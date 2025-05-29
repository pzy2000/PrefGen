from __future__ import absolute_import, division, print_function
import sys
sys.path.append('..')
import argparse
import json
import os
import time
import warnings
from datetime import datetime
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logger import MyLogger
from utils.retrieve import init_bert_model, query



def few_shot_generation(args, prompt, tokenizer, model, sample):
    output_list = []
    if args.model == "DeepSeek-V3":
        client = OpenAI(api_key="sk-b953520bbda24a11b6d9af3574d83ad1", base_url="https://api.deepseek.com/beta")
        for _ in range(sample):
            response = client.completions.create(
                model="deepseek-chat",
                prompt=prompt,
                max_tokens=1024,
                # top_p=args.p,
                temperature=args.temperature,
                suffix="// END_OF_FUNCTION",
            )
            output = response.choices[0].text
            logger.info_blue(prompt + output)
            logger.info_blue("=====================================")
            output_list.append(output)
    elif args.model == "Qwq-32b":
        client = OpenAI(api_key="sk-0325f9d7eab8485198303f6329726a3b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        for _ in range(sample):
            response = client.chat.completions.create(
                model="qwq-32b",
                messages=[
                    {"role": "system",
                     "content": "You are a professional Solidity engineer. Please continue to generate a function based on the provided requirement and function signature, NO need to repeat the signature. End your function with // END_OF_FUNCTION, and Code ONLY"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                stream=True,
                # top_p=args.p,
                temperature=args.temperature,
                # suffix="// END_OF_FUNCTION",
            )
            print("\n" + "=" * 20 + "Thinking process" + "=" * 20 + "\n")
            reasoning_content = ""
            answer_content = ""
            is_answering = False
            for chunk in response:
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                        print(delta.reasoning_content, end='', flush=True)
                        reasoning_content += delta.reasoning_content
                    else:
                        if delta.content != "" and is_answering is False:
                            print("\n" + "=" * 20 + "Full reply" + "=" * 20 + "\n")
                            is_answering = True
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content
            output = answer_content
            # output = output[output.find("```solidity"):]
            output = output[:output.rfind("// END_OF_FUNCTION")]
            # output = output.replace("```solidity", "")
            logger.info_blue(output)
            logger.info_blue("=====================================")
            output_list.append(output)
    # elif args.model == "DeepSeek-R1":
    #     client = OpenAI(api_key="sk-b953520bbda24a11b6d9af3574d83ad1", base_url='https://api.deepseek.com/beta')
    #     for _ in range(sample):
    #         response = client.chat.completions.create(
    #             model="deepseek-reasoner",
    #             messages=[
    #                 {"role": "system", "content": "You are a professional Solidity engineer. Please continue to generate a function based on the provided requirement and function signature, NO need to repeat the function signature. End your function with // END_OF_FUNCTION, and Code ONLY"},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             max_tokens=1024,
    #             # top_p=args.p,
    #             temperature=args.temperature,
    #             # suffix="// END_OF_FUNCTION",
    #         )
    #         output = str(response.choices[0].message.content)
    #         # output = output[output.find("```solidity"):]
    #         # output = output[:output.rfind("// END_OF_FUNCTION")]
    #         # output = output.replace("```solidity", "")
    #         logger.info_blue(prompt + output)
    #         logger.info_blue("=====================================")
    #         output_list.append(output)
    elif args.model == "gpt-4o-mini":
        client = OpenAI(api_key="fk230647-IN0mMHTRLndX59NxjHI0FZ0zlvPtr39C",
                        base_url="https://oa.api2d.net")
        for _seed in range(sample):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer",
                     "content": "You are a professional Solidity engineer. Please continue to generate a function based on the provided requirement and function signature, NO need to repeat the signature. End your function with // END_OF_FUNCTION"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                # n=10,
                temperature=args.temperature,
                seed=_seed

            )
            output = str(response.choices[0].message.content)
            output = output[:output.rfind("// END_OF_FUNCTION")]
            output = output.replace("```solidity", "")
            output = function_full_sig.strip('\n') + '\n' + output.strip('\n')
            logger.info_blue(output)
            # for idx, out in enumerate(output_list):
            with open(
                    f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt",
                    'w') as f:
                f.write(output)
    elif args.model == "gpt-4o":
        client = OpenAI(api_key="fk230647-IN0mMHTRLndX59NxjHI0FZ0zlvPtr39C",
                        base_url="https://oa.api2d.net")
        for _seed in range(sample):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "developer",
                     "content": "You are a professional Solidity engineer. Please continue to generate a function based on the provided requirement and function signature, NO need to repeat the signature. End your function with // END_OF_FUNCTION"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                # n=10,
                temperature=args.temperature,
                seed=_seed
            )
            output = str(response.choices[0].message.content)
            output = output[:output.rfind("// END_OF_FUNCTION")]
            output = output.replace("```solidity", "")
            output = function_full_sig.strip('\n') + '\n' + output.strip('\n')
            logger.info_blue(output)
            with open(
                    f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx}.txt",
                    'w') as f:
                f.write(output)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        raw_outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            # top_p=args.p,
            # top_k=args.k,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=sample
        )
        for raw_output in raw_outputs:
            logger.info_blue(tokenizer.decode(raw_output))
            output = tokenizer.decode(raw_output[len(inputs[0]):])
            if output.find("// END_OF_FUNCTION"):
                output = output[:output.find("// END_OF_FUNCTION")].rstrip()
                output = output[:output.rfind("}") + 1]
            output_list.append(output)
    return output_list


if __name__ == '__main__':
    with open('data/new_dataset_0410.json', 'r') as file:
        data = json.load(file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    total_inference_time = 0
    inference_tries = 0
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="CodeLlama_7b")
    parser.add_argument('--k',
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. '
                             'Only applies for sampling mode, with range from 1 to 100.',
                        type=int, default=50)
    parser.add_argument('--p',
                        help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. '
                             'The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. '
                             'Only applies to sampling mode. Also known as nucleus sampling.',
                        type=float, default=0.95)
    parser.add_argument('--temperature',
                        help='What sampling temperature to use, between 0 and 2. '
                             'Higher values like 0.8 will make the output more random, '
                             'while lower values like 0.2 will make it more focused and deterministic.',
                        type=float, default=1)
    parser.add_argument('--context', action='store_true', default=False, help='Enable context for generation')
    parser.add_argument('--testcase', action='store_true', default=False, help='Enable testcase for generation')
    parser.add_argument('--shot', help='', type=int, default=2)
    args = parser.parse_args()
    embedding_list, original_document_list, func_list = init_bert_model()
    log_file = f"log_{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}_{current_time}.txt"
    logger = MyLogger(f"logs_patch/rag/{log_file}")
    total_token = 0
    token_tries = 0
    if args.model == "DeepSeek-Coder-33B":
        tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct',
                                                  use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-33b-instruct',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    elif args.model == "CodeLlama_7b":
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-7b-Instruct-hf',
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "CodeLlama-34B":
        tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-34b-Instruct-hf', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-34b-Instruct-hf',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    elif args.model == "Qwen-7B":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    elif args.model == "Qwen-7B-dpo":
        tokenizer = AutoTokenizer.from_pretrained("/root/.cache/huggingface/models/checkpoint-2267", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('/root/.cache/huggingface/models/checkpoint-2267',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='auto')
    elif args.model == "DeepSeek-R1-Distill-Llama-8B":
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                     device_map='cuda')

    elif args.model == "Qwen-32B":
        tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-Coder-32B-Instruct',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-Coder-32B-Instruct',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "DeepSeek-Coder":
        tokenizer = AutoTokenizer.from_pretrained(
            'deepseek-ai/deepseek-coder-6.7b-instruct',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'deepseek-ai/deepseek-coder-6.7b-instruct',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "CodeLlama":
        tokenizer = AutoTokenizer.from_pretrained(
            'codellama/CodeLlama-34b-Instruct-hf',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'codellama/CodeLlama-34b-Instruct-hf',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "DeepSeek-Coder-V2":
        tokenizer = AutoTokenizer.from_pretrained(
            'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "Magicoder":
        tokenizer = AutoTokenizer.from_pretrained(
            'ise-uiuc/Magicoder-S-DS-6.7B',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'ise-uiuc/Magicoder-S-DS-6.7B',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "OpenCode":
        tokenizer = AutoTokenizer.from_pretrained(
            'm-a-p/OpenCodeInterpreter-DS-6.7B',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'm-a-p/OpenCodeInterpreter-DS-6.7B',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "OpenCode-33B":
        tokenizer = AutoTokenizer.from_pretrained(
            'm-a-p/OpenCodeInterpreter-DS-33B',
            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            'm-a-p/OpenCodeInterpreter-DS-33B',
            trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "KwaiCoder":
        tokenizer = AutoTokenizer.from_pretrained('Kwaipilot/KwaiCoder-23B-A4B-v1', use_fast=True)
        model = AutoModelForCausalLM.from_pretrained('Kwaipilot/KwaiCoder-23B-A4B-v1', trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "QwQ-32b":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/QwQ-32B", trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map='auto')
    elif args.model == "DeepSeek-R1":
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True,
                                                      device_map='auto')
    # elif args.model == "DPO":
    #     model = AutoModelForCausalLM.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-DPO",
    #                                                  trust_remote_code=True,
    #                                                  device_map='auto')
    #     tokenizer = AutoTokenizer.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-DPO", use_fast=True)
    # elif args.model == "SFT":
    #     model = AutoModelForCausalLM.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT",
    #                                                  trust_remote_code=True,
    #                                                  device_map='auto')
        # tokenizer = AutoTokenizer.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT", use_fast=True)
    elif args.model == "Qwen-7B-SFT-DPO":
        # /home/dataset/GreenCoder/saved_models/Qwen-7B-SFT-DPO
        model = AutoModelForCausalLM.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT-DPO",
                                                     trust_remote_code=True,
                                                     device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT-DPO", use_fast=True)
    elif args.model == "Qwen-7B-SFT":
        # /home/dataset/GreenCoder/saved_models/Qwen-7B-SFT-DPO
        model = AutoModelForCausalLM.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT",
                                                     trust_remote_code=True,
                                                     device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained("/home/dataset/GreenCoder/saved_models/Qwen-7B-SFT", use_fast=True)
    else:
        tokenizer, model = None, None

    if not os.path.exists(f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}"):
        os.makedirs(f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}")

    import pickle

    real_path_cargo = pickle.load(open("prebuilt/real_path_cargo.pkl", "rb"))
    for file_path, file_contents in tqdm(data.items(), colour='green'):
        logger.info_white("file_path:\n" + file_path)
        file_path = file_path.replace("/root/", "repository/")
        for file_content in file_contents:
            for method in tqdm(file_content['methods'], colour="red"):
                if 'llm_comment' not in method.keys():
                    continue
                if file_path not in real_path_cargo.keys():
                    continue
                # real_file_path = file_path
                identifier = method['identifier']
                if os.path.exists(
                        f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{args.sample - 1}.txt"):
                    logger.info_blue(
                        f"exist patch file, skipping {real_path_cargo[file_path].split('/')[-1]}_function_{identifier}")
                    continue

                comment = method['llm_comment'].strip()
                context = eval(method['context']) if method['context'] != "" else None
                # if context == set():
                # context = None
                function_full_sig = method['full_signature'].strip() + ' {' + '\n'

                prompt = ""
                examples = query(comment, embedding_list, original_document_list, func_list, args.shot)
                for example in examples:
                    prompt = prompt + "// IMPLEMENT THE FUNCTIONALITY BASED ON THE PROVIDED REQUIREMENT.\n\n// START_OF_REQUIREMENT\n" + \
                            example[0] + "\n// END_OF_REQUIREMENT\n\n" + "// START_OF_CONTEXT" + '\n' + example[
                                2] + "\n// END_OF_CONTEXT" + '\n\n' + "// START_OF_FUNCTION\n" + example[
                                1] + "\n// END_OF_FUNCTION\n\n"
                prompt = prompt + "// IMPLEMENT THE FUNCTIONALITY BASED ON THE PROVIDED REQUIREMENT.\n\n// START_OF_REQUIREMENT\n" + comment + "\n// END_OF_REQUIREMENT\n"

                if args.context:
                    prompt = prompt + '\n' + "// START_OF_CONTEXT" + '\n'
                    if not context:
                        prompt = prompt + "No context for this function" + '\n'
                    else:
                        for c in context:
                            prompt = prompt + c + '\n'
                    prompt = prompt + "// END_OF_CONTEXT" + '\n'
                prompt = prompt + '\n' + "// START_OF_FUNCTION" + '\n' + function_full_sig
                import glob
                import os
                import re

                have_sample = 0
                pattern = f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_*"
                matching_files = []
                for file_path_ in glob.glob(pattern):
                    if os.path.isfile(file_path_) and re.search(rf'_{identifier}_\d+\.txt$', file_path_):
                        matching_files.append(file_path_)
                have_sample = len(matching_files)
                # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                # if len(inputs[0]) > 2048:
                #     continue
                # while True:
                try:
                    start_time = time.time()
                    output_list = few_shot_generation(args, prompt, tokenizer, model, args.sample - have_sample)
                    end_time = time.time()
                    inference_tries += args.sample - have_sample
                    total_inference_time += end_time - start_time
                    average_inference_time = total_inference_time / inference_tries
                    logger.info_green("average_inference_time: {:.2f}s".format(average_inference_time))
                    output_list = [function_full_sig.strip('\n') + '\n' + output.strip('\n') for output in
                                output_list] if args.model != "Qwq-32b" \
                        else [output.strip('\n') for output in output_list]
                    if args.model != "gpt-4o-mini":
                        for idx, out in enumerate(output_list):
                            with open(
                                    f"patch/rag/{args.model}_shot_{args.shot}_context_{args.context}_testcase_{args.testcase}/patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}_{idx + have_sample}.txt",
                                    'w') as f:
                                f.write(out)
                    # break
                except Exception as e:
                    logger.error(e)
                    if "Too Many Requests for your account" in str(e):
                        for _ in tqdm(range(120), desc="sleeping", colour="yellow"):
                            time.sleep(1)
                            # slow = True
