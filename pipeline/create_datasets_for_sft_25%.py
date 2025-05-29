import json
import random
import os
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
# from utils.retrieve import init_bert_model, query
random.seed(42)


def get_solidity_distilled_data(data_path):
    """
    处理自制 Solidity 代码生成数据集：
      - 读取 all_results.json 和 all_codes.json 文件（结构详见说明）。
      - 遍历每个 patch（即每个问题或函数），直接从 all_results 获取 gas fee 的最高和最低版本候选代码。
      - 基于最高版本候选（max_）构造 prompt（利用 llm_comment 与 full_signature），
        并将 max_ 和 min_ 中的 body 分别作为 response_j 与 response_k。
      - 使用 Solidity 代码块标记（```solidity）包装代码。
    """
    # all_results = json.load(open(f"data/test_data.json"))
    all_results_train = json.load(open(f"{data_path}/all_results.jsonl"))
    output_file = f"{data_path}/distilled_solidity_data_25%.json"
    with open(output_file, "w") as file:
        # for patch_id, patch_data in tqdm(all_results.items(), total=len(all_results)):
        for patch_id, patch_data in tqdm(all_results_train.items(), total=len(all_results_train)):
            max_item = patch_data["max_"]
            min_item = patch_data["min_"]

            llm_comment = max_item.get("llm_comment", "").rstrip()
            full_signature = max_item.get("patch", "").rstrip()[0:max_item.get("patch", "").rstrip().find("\n")].replace("{", "")

            prompt = "// IMPLEMENT THE FUNCTIONALITY BASED ON THE PROVIDED REQUIREMENT.\n\n// START_OF_REQUIREMENT\n" + \
                         llm_comment + "\n// END_OF_REQUIREMENT\n\n" + "// START_OF_FUNCTION\n" + full_signature
            # exit()

            sol_max = max_item.get("patch", "").rstrip()
            sol_min = min_item.get("patch", "").rstrip()
            file_pat = max_item.get("file_path")
            
            real_file_pat = max_item.get("file_path")
            Pass = max_item.get("PASS")
            if Pass != "True":
                continue
            Pass2 = min_item.get("PASS")
            if Pass2 != "True":
                continue
            COMPILE_Pass = max_item.get("COMPILE_PASS")
            Id = max_item.get("identifier")
            task_id = file_pat + "_" + Id
            model_name = min_item.get("model_name")

            response_j = f"```solidity\n{sol_max}\n```"
            response_k = f"```solidity\n{sol_min}\n```"

            
            if response_j == response_k == "```solidity\nNone\n```":continue
            if response_k == "```solidity\nNone\n```": continue
            func_content = max_item
            total_max_gas = 0
            for test_func in func_content['GAS'].keys():
                gas_0 = int(func_content['GAS'][test_func]['gas']) if func_content['GAS'][test_func]['gas'] != 'None' else None
                gas_1 = int(func_content['GAS'][test_func]['~']) if func_content['GAS'][test_func]['~'] != 'None' else None
                gas_2 = int(func_content['GAS'][test_func]['μ']) if func_content['GAS'][test_func]['μ'] != 'None' else None
                total_max_gas += (gas_0 if gas_0 else 0) + (gas_1 if not gas_0 else 0) + (gas_2 if not gas_0 else 0)

            func_content = min_item
            total_min_gas = 0
            for test_func in func_content['GAS'].keys():
                gas_0 = int(func_content['GAS'][test_func]['gas']) if func_content['GAS'][test_func]['gas'] != 'None' else None
                gas_1 = int(func_content['GAS'][test_func]['~']) if func_content['GAS'][test_func]['~'] != 'None' else None
                gas_2 = int(func_content['GAS'][test_func]['μ']) if func_content['GAS'][test_func]['μ'] != 'None' else None
                total_min_gas += (gas_0 if gas_0 else 0) + (gas_1 if not gas_0 else 0) + (gas_2 if not gas_0 else 0)

            # if total_max_gas != 0 and (total_max_gas - total_min_gas) / total_max_gas < 0.1:
                # continue
            ratio = (total_max_gas - total_min_gas) / total_max_gas if total_max_gas != 0 else -100
            sample_dict = {
                "task_id": task_id,
                "prompt": prompt,
                "response_j": response_j,
                "response_k": response_k,
                "file_path": file_pat,
                "real_file_path": real_file_pat,
                "PASS": Pass,
                "COMPILE_PASS": COMPILE_Pass,
                "identifier": Id,
                "Model_name": model_name,
                "patch_id": patch_id,
                "total_max_gas": total_max_gas,
                "total_min_gas": total_min_gas,
                "ratio": ratio
            }
            file.write(json.dumps(sample_dict) + "\n")
    print(f"Solidity distilled data saved to {output_file}")



def process_data_for_sft(sample):  
    messages = [
        {"role": "user", "content": sample["prompt"].rstrip()},
        {
            "role": "assistant", 
            "content": sample["response_k"].rstrip()
        },
    ]
    return {"messages": messages}

def process_data_for_dpo(sample):
    return {
        "prompt": sample["prompt"].rstrip(),
        "chosen": [
            { 
                "role": "user",
                "content": sample["prompt"].rstrip(),
            },
            {
                "role": "assistant",
                "content": sample["response_k"].rstrip(),
            }
        ],
        "rejected": [
            { 
                "role": "user",
                "content": sample["prompt"].rstrip(),
            },
            {
                "role": "assistant",
                "content": sample["response_j"].rstrip(),
            }
        ],
        # "PASS": sample["PASS"],
        # "COMPILE_PASS": sample["COMPILE_PASS"],
        # "file_path": sample["file_path"],
        # "real_file_path": sample["real_file_path"],
        # "identifier": sample["identifier"],
        # "Model_name": sample["Model_name"]
    }
    
def process_data_for_grpo(sample):  
    return {
        "prompt": [
            {"role": "user", "content": sample["prompt"].rstrip()}
        ],
        "completion": [
            {
                "role": "assistant", 
                "content": sample["response_k"].rstrip()
            }
        ]
    }

def process(num_proc: int, data_path: str, post_training_type: str):
    """
    此函数加载 distilled 数据（针对 Solidity 数据集使用 distilled_solidity_data.json），
    并基于不同后训练类型（sft, dpo, grpo）对数据进行包装，同时保留原始所有 key-value 信息，
    然后对数据进行 train/test 切分，并将两个 split 分别以 JSON 格式保存。
    """

    dataset = load_dataset('json', data_files=f"{data_path}/distilled_solidity_data_25%.json")["train"]
    dataset = dataset.shard(num_shards=4, index=0)
    from functools import partial
    if post_training_type == "sft":
        process_example_map = partial(process_data_for_sft)
    elif post_training_type == "dpo":
        process_example_map = partial(process_data_for_dpo)
    elif post_training_type == "grpo":
        process_example_map = partial(process_data_for_grpo)
    original_columns = dataset.column_names
    import copy
    original_dataset = copy.deepcopy(dataset)

    dataset = dataset.map(
        process_example_map, 
        num_proc=num_proc,
        desc="Tokenizing data",
        remove_columns=original_columns
    )


    dataset_split = dataset.train_test_split(test_size=1/9, shuffle=True, seed=42)
    original_dataset_split = original_dataset.train_test_split(test_size=1/9, shuffle=True, seed=42)

    train_data = list(original_dataset_split["train"])
    test_data = list(original_dataset_split["test"])
    

    train_save_path = f"./datasets/{post_training_type}_solidity_train.json"
    test_save_path = f"./datasets/{post_training_type}_solidity_test.json"
    

    with open(train_save_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_save_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Train split saved to {train_save_path}")
    print(f"Test split saved to {test_save_path}")
    
    return dataset_split


if __name__ == "__main__":
    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
    # embedding_list, original_document_list, func_list = init_bert_model()
    data_path = "/home/pdia/data/GreenCoder/results/solidity"

    get_solidity_distilled_data(data_path)
    

    for post_training_type in ["sft"]:        
        cache_dir = f"./datasets/{post_training_type}_solidity_data_25%"
        dataset_proc = process(num_proc=20, data_path=data_path, post_training_type=post_training_type)
        dataset_proc.save_to_disk(cache_dir)
