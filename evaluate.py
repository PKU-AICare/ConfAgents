import os
import json
import numpy as np
import argparse
from utils.json_utils import preprocess_response_string
from typing import Union, Optional

def bootstrap(data_path):
    """
    Bootstrap the sample qids with replacement, the size of the sample is the same as the original data.
    """
    data_qids = []
    with open(data_path, "r") as f:
        data = json.load(f)
        for datum in data:
            data_qids.append(datum["qid"])
            
    return [data_qids[i] for i in np.random.randint(0, len(data_qids), len(data_qids))]

def extract_digits(string):
    """
    Extract digits from a string.
    """
    return "".join(filter(str.isdigit, string))

def add_llm_score_to_json(json_file_path: str, llm_score: str):
    """
    Reads a JSON file, adds an 'llm_score' key at the top level, 
    and saves the modified data back to the original file path.

    Args:
        json_file_path: The path to the JSON file to modify.
        llm_score: The score string to add under the 'llm_score' key.

    Raises:
        FileNotFoundError: If the specified json_file_path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        Exception: For other potential I/O or unexpected errors.
    """
    try:
        # 1. Read the existing JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Add the llm_score key-value pair
        # Ensure it's treated as a string if that's the requirement,
        # otherwise, you might want to convert llm_score to int/float earlier.
        data['llm_score'] = llm_score 

        # 3. Write the modified data back to the original file
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # Use indent for readability; ensure_ascii=False for wider char support
            json.dump(data, f, indent=2, ensure_ascii=False) 
            
        # print(f"Successfully added 'llm_score' to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: File not found at '{json_file_path}'")
        raise # Re-raise the exception if you want the caller to handle it
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{json_file_path}'. Is it a valid JSON file?")
        raise # Re-raise
    except Exception as e:
        print(f"An unexpected error occurred while processing '{json_file_path}': {e}")
        raise # Re-raise

def extract_score_from_llm_output(output_string: str) -> Union[int,None]:
    """
    Extracts the first integer value associated with the key "score" 
    from a potentially malformed JSON-like string output by an LLM.

    It specifically looks for '"score":' followed by optional whitespace 
    and then an integer. It does not rely on full JSON parsing.

    Args:
        output_string: The string output from the LLM, expected to contain 
                       a '"score": <number>' pattern.

    Returns:
        The extracted integer score if found, otherwise None.
    """
    if not output_string:
        return None

    # Option 1: Using string manipulation (more step-by-step)
    try:
        # Find the position of '"score":'
        score_key = '"score":'
        key_index = output_string.find(score_key)
        
        if key_index == -1:
            # Try with single quotes as a fallback, as LLMs might hallucinate them
            score_key = "'score':" 
            key_index = output_string.find(score_key)
            if key_index == -1:
                return None # Key not found

        # Start searching for the number right after '"score":'
        start_search_index = key_index + len(score_key)
        
        # Skip any whitespace characters immediately after the colon
        num_start_index = start_search_index
        while num_start_index < len(output_string) and output_string[num_start_index].isspace():
            num_start_index += 1
            
        if num_start_index == len(output_string):
            return None # Reached end of string without finding a number

        # Extract consecutive digits
        num_end_index = num_start_index
        while num_end_index < len(output_string) and output_string[num_end_index].isdigit():
            num_end_index += 1
            
        # If no digits were found right after skipping whitespace
        if num_end_index == num_start_index:
            return None 
            
        # Extract the number string and convert to int
        number_str = output_string[num_start_index:num_end_index]
        return int(number_str)

    except Exception:
        # Catch any unexpected errors during string processing
        return None

if __name__ == "__main__":
    np.random.seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="./output")
    parser.add_argument("--bootstrap", type=bool, default=True, help="Whether to bootstrap the data")
    parser.add_argument("--n_bootstrap", type=int, default=10, help="Number of bootstrap samples")
    parser.add_argument("--model", type=str, default="deepseek-v3-official")
    args = parser.parse_args()
    
    # Loop through the datasets
    for dataset_dir in os.listdir(logs_dir):
        if not os.path.isdir(os.path.join(logs_dir, dataset_dir)):
            continue
        
        dataset = dataset_dir
        print(f"Dataset: {dataset}")
        
        # Loop through the question types
        
        data_path = f"./data/{dataset}/processed.json"
        
        qids = []
        MODEL_WIDTH = 17
        METRIC_WIDTH = 10
        VALUE_WIDTH = 8 
        TOTAL_WIDTH = 8
        TIME_WIDTH = 8
        
        print(f"{'Model':<{MODEL_WIDTH}} | {'Metric':<{METRIC_WIDTH}} | {'Mean':>{VALUE_WIDTH}} | {'Std':>{VALUE_WIDTH}} | {'Total':>{TOTAL_WIDTH}} | {'Time':>{TOTAL_WIDTH}} | {'Time_Std':>{TOTAL_WIDTH}} | {'C_Token':>{TOTAL_WIDTH}} | {'T_Token':>{TOTAL_WIDTH}}"  )
        print(f"{'-' * MODEL_WIDTH}-+-{'-' * METRIC_WIDTH}-+-{'-' * VALUE_WIDTH}-+-{'-' * VALUE_WIDTH}-+-{'-' * TOTAL_WIDTH}-+-{'-' * TOTAL_WIDTH}-+-{'-' * TOTAL_WIDTH}-+-{'-' * TOTAL_WIDTH}-+-{'-' * TOTAL_WIDTH}")
        
        model_order = [""] # ADD YOUR MODELS
        
        np.random.seed(42)

        # 生成所有bootstrap样本
        qids = [bootstrap(data_path) for _ in range(args.n_bootstrap)]
        
        # Loop through the model results
        for model_dir in model_order:
            if model_dir in os.listdir(os.path.join(logs_dir, dataset)):
                if not os.path.isdir(os.path.join(logs_dir, dataset, model_dir)):
                    continue
                
                model = model_dir
                result = {"model": model, "acc": [], "score": [], "total": 0, "avg_time":[], "token":[],"total_token":[]}
                
                # Loop through each bootstrap sample
                for i in range(len(qids)):
                    correct = 0
                    time = 0
                    token = 0
                    total = 0
                    total_token = 0
                    for qid in qids[i]:
                        for ans_file in os.listdir(os.path.join(logs_dir, dataset,model)):
                            if extract_digits(str(qid)) == extract_digits(ans_file):
                                total += 1
                                try:
                                    ans_data = json.load(open(os.path.join(logs_dir, dataset,model,ans_file), "r"))
                                except Exception as e:
                                    print(f"Error loading {tmp_path}: {e}")
                                    continue

                                if ans_data["ground_truth"] == ans_data["predicted_answer"]:
                                    correct += 1
                                
                                time += ans_data['case_history']['processing_time']
                                token += ans_data['case_history']['completion_tokens']
                                total_token += ans_data['case_history']['total_tokens']
                                        
                    result["acc"].append(correct / total)
                    result["total"] = total
                    result["token"].append(token/total)
                    result['avg_time'].append(time / total)
                    result['total_token'].append(total_token / total)
                
                metric_name = "Accuracy"
                mean_value = round(np.mean(result["acc"]), 4)
                std_dev = round(np.std(result["acc"]), 4)
                total_str = str(result['total'])
                mean_time = round(np.mean(result["avg_time"]), 4)
                std_time = round(np.std(result["avg_time"]), 4)
                tokens = round(np.mean(result["token"]), 4)
                total_tokens = round(np.mean(result["total_token"]), 4)
                
                print(f"{model:<{MODEL_WIDTH}} | {metric_name:<{METRIC_WIDTH}} | {mean_value:>{VALUE_WIDTH}.4f} | {std_dev:>{VALUE_WIDTH}.4f} | {total_str:>{TOTAL_WIDTH}} | {mean_time:>{TOTAL_WIDTH}} | {std_time:>{TOTAL_WIDTH}} | {token:>{TOTAL_WIDTH}} | {total_token:>{TOTAL_WIDTH}}")
