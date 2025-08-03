import os
import json
import time
import argparse
from tqdm import tqdm
import asyncio
from asyncio import Semaphore
from tqdm.asyncio import tqdm

# Import ColaCare utilities
from utils.json_utils import load_json, save_json, preprocess_response_string
from utils.encode_image import encode_image

from agents.llm_configs import LLM_MODELS_SETTINGS
from agents.assistant_agent import *
from agents.base_agent import *
from agents.conformal_agent import *
from agents.framework import *
from agents.main_agent import *
from agents.prompt import * 

def read_jsonl(path, default_size=200):
    json_dict = {}
    json_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # 去除换行符并加载每一行的 JSON
            meta_info = json_obj["meta_info"]
            if meta_info not in json_dict:
                json_dict[meta_info] = []
            json_list.append(json_obj)
            json_dict[meta_info].append(json_obj)

    for meta_info in json_dict:
        assert len(json_dict[meta_info]) == default_size
        
    return json_list, json_dict
import contextvars
import functools
async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)

async def process_data_parallel(data, max_concurrent=5):
    """并行处理数据，限制最大并发数"""
    semaphore = Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            return await to_thread(process_item, item)
    
    tasks = [process_with_semaphore(item) for item in data]
    
    # 保持顺序的并行处理
    from tqdm.asyncio import tqdm
    results = await tqdm.gather(*tasks, desc="Processing")
    
    return results

def process_item(item):
    
    agent_configs = {
        'main': {
            "model_key": args.model,
            "temperature": 1
        },
        'conformal': {
            "model_key": "conformal_predictor",
            "alpha": args.alpha
        },
        'assistant': [
            {"model_key": args.model, "temperature": 0.5},
            {"model_key": args.model, 'temperature': 0.5},
            {"model_key": args.model, 'temperature': 0.5},
        ]
    }
    
    framework = Framework(agent_configs)
    start_time = time.time()
    qid = item.get("qid")
    result_path_refined = os.path.join(logs_dir, f"{qid}-result.json")
    
    if os.path.exists(result_path_refined):
        print(result_path_refined)
        print(f"Skipping {qid} (already processed)", flush=True)
        return None

    res = framework.process_debate(item, is_rag=args.is_rag,ablation_type=args.ablation_type)

    print(f"QID:{qid}, Initial Answer: {res['pred_answer']}, Refined Answer: {res['refined_answer']}, Confidence_level: {res['confidence_level']}, Target: {item['answer']}")

    processing_time = time.time() - start_time

    result_json_refined = {
        'qid': qid,
        "timestamp": int(time.time()),
        "question": item['question'],
        "options": item['options'],
        "image_path": None,
        "ground_truth": item['answer'],
        "initial_answer": res["pred_answer"],
        "predicted_answer": res["refined_answer"],
        "case_history": {
            "processing_time": processing_time,
            "confidence_level":res["confidence_level"],
            "initial_judgement": res["initial_judgement"],
            "assist_judgement": res["assist_infos"],
            "refined_judgement": res["final_judgement"],
            "conformal_size": len(res["pset_infos"]),
            "conformal_set": res["pset_infos"],
            "rag_infos": res["rag_infos"],
            "total_tokens": res["total_tokens"],
            "completion_tokens": res["completion_tokens"]
        }
    }
    save_json(result_json_refined, result_path_refined)
    return result_json_refined

def main():
    """
    Main entry point for running the Reconcile framework from command line.
    """
    
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on medical QA datasets")
    parser.add_argument("--dataset", type=str, choices=["medbullets","medqa","medmcqa","mmlu","VQA-RAD","pubmedqa","afrimedqa","medexqa","medxpertqa-r"])
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of discussion rounds")
    parser.add_argument("--is_rag", type=bool, default=True)
    parser.add_argument("--model", type=str, choices=["deepseek-v3-official", "gpt-4o","gpt-4o-mini","gpt-o4-mini","qwen2.5-72b-instruct","kimi-k2","deepseek-r1-official","claude-sonnet-4"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--split", type=str,choices=['hard',"easy","ablation","all"],help="Directory to save output files")
    parser.add_argument("--ablation_type", type=str, default="",choices=['','rag',"cp","network","sens"])
    global args, logs_dir, framework, size_filtered_data
    args = parser.parse_args()
    method = "Conformal"
    print(args.is_rag)

    # Extract dataset name
    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model}")
    if args.split == 'hard':
        logs_dir = os.path.join("output-hard", args.model, dataset_name, method)
        os.makedirs(logs_dir, exist_ok=True)
        data_path = f"./data/{dataset_name}/processed_50_hard.json"
    elif args.split == 'easy':
        logs_dir = os.path.join("output-simple", args.model, dataset_name, method)
        os.makedirs(logs_dir, exist_ok=True)
        data_path = f"./data/{dataset_name}/processed_50.json"
    elif args.split == 'all':
        logs_dir = os.path.join("output", args.model, dataset_name, method)
        os.makedirs(logs_dir, exist_ok=True)
        data_path = f"./data/{dataset_name}/processed.json"
    elif args.split == 'ablation':
        logs_dir = os.path.join("output-ablation", args.model, dataset_name, method, f"{args.ablation_type}_{args.alpha}")
        os.makedirs(logs_dir, exist_ok=True)
        data_path = f"./data/{dataset_name}/processed.json"
            
    # Construct the data path

    # Load the dataset
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    asyncio.run(process_data_parallel(data, max_concurrent=5))

if __name__ == "__main__":
    main()