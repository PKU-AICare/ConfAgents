import os
import json
from collections import defaultdict

def process_folder(root_folder):
    
    # 遍历根文件夹下的所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        all_data = []
        qid_counter = 1  # 从1开始编号
        name = dirpath.split('/')[-1]

        # exit()
        # 检查当前文件夹中是否有sampled_50_hard.jsonl文件
        if 'sampled_50.jsonl' in filenames:
            file_path = os.path.join(dirpath, 'sampled_50.jsonl')
            
            # 读取jsonl文件
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取所需字段并重命名answer_idx
                        processed_data = {
                            "qid": qid_counter,
                            "question": data.get("question", ""),
                            "options": data.get("options", []),
                            "answer": data.get("answer_idx", -1)  # 默认值-1如果字段不存在
                        }
                        
                        all_data.append(processed_data)
                        qid_counter += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
    
        # 保存处理后的数据
        output_path = os.path.join(dirpath, 'processed.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    
        print(f"处理完成！共处理{qid_counter-1}条数据，结果已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    folder_path = "" # ADD PATH
    process_folder(folder_path)