from .main_agent import * 
from .conformal_agent import * 
from .assistant_agent import * 
import os 

class Framework:
    def __init__(self, agents_config: Dict[str, Any]):
        print("initializing framework...")
        self.main_agent = MainAgent(
            model_key=agents_config['main']["model_key"],
            temperature=agents_config['main']["temperature"]
        )
        self.conformal_agent = ConformalAgent(
            model_key=agents_config['conformal']["model_key"],
            alpha=agents_config['conformal']["alpha"]
        )
        self.assistants = [
            AssistAgent(
                model_key=agents_config['assistant'][i]["model_key"],
                temperature=agents_config['assistant'][i]["temperature"]
            ) for i in range(len(agents_config['assistant']))
        ]
            
    def extract_pset(self, pset, options):
        mapping = {0:'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        pset_infos = []
        for i in range(len(options)):
            if pset[i]:
                pset_info = {
                    'option': mapping[i],
                    'content': options[mapping[i]]
                }
                pset_infos.append(pset_info)
        return pset_infos

    def process_debate(self, info, medical_field=None, is_rag = False, ablation_type=""):
        """
        Process the information using the agent's initial judgement method.
        
        Args:
            info: Information block to process
            medical_field: Optional medical field for context
            
        Returns:
            Tuple indicating success and the result dictionary
        """

        # step 1: get the initial judgement based on the information block
        _, initial_judgement = self.main_agent.initial_judgement(info, medical_field)

        # step 2: process the initial judgement using the conformal agent
        logits = initial_judgement['logits']
        confidences = initial_judgement['confidence']
        pset = self.conformal_agent.conformal_prediction(logits)
        pset_infos = self.extract_pset(pset, info['options'])
        # print(logits,pset_infos)

        # step 3: assistant helper
        assist_infos = []
        print(f"Calling {len(pset_infos)} assistants for further analysis...")

        
        if len(pset_infos)==1:
            res = {
                "initial_judgement":initial_judgement,
                "final_judgement":"Confident for the answer, no need for further judgement",
                "assist_infos":[],
                "pred_answer":initial_judgement['pred_answer'],
                "refined_answer":initial_judgement['pred_answer'],
                "pset_infos":pset_infos,
                "confidence_level":"High",
                "rag_infos":[],
                "total_tokens": self.main_agent.total_tokens,
                "completion_tokens": self.main_agent.completion_tokens
            }
            return res 
        
        rag_infos=[]
        
        for i in range(min(len(pset_infos),3)):
            res,rag_infos = self.assistants[i].assist_info_debate(info)
            assist_infos.append(res)

        # step 4: final judgement
        refined_answer,final_judgement = self.main_agent.refined_judgement_debate(info,initial_judgement['pred_answer'],assist_infos)

        if refined_answer not in ['A', 'B', 'C', 'D', 'E']:
            print(f"Refined answer {refined_answer} is not valid, using initial judgement answer instead.")
            refined_answer = initial_judgement['pred_answer']

        total_token = self.main_agent.total_tokens + sum([assistant.total_tokens for assistant in self.assistants])
        completion_tokens = self.main_agent.completion_tokens + sum([assistant.completion_tokens for assistant in self.assistants])
        print(total_token, completion_tokens)
        
        res = {
            "initial_judgement":initial_judgement,
            "final_judgement":final_judgement,
            "assist_infos":assist_infos,
            "pred_answer":initial_judgement['pred_answer'],
            "refined_answer":refined_answer,
            "pset_infos":pset_infos,
            "confidence_level":final_judgement['confidence_level'],
            "rag_infos":rag_infos,
            "total_tokens": total_token,
            "completion_tokens": completion_tokens
        }
        return res

    def run(self,meta_infos, json_dict, output_dir, psets_json=None):
        results = {}
        for i in range(len(meta_infos)): 
            print(f"{i}: {meta_infos[i]}")   
            results[meta_infos[i]]={'initial_answer': [], 'refined_answer': [], 'scores': [], 'targets': []}
            for j in range(len(json_dict[meta_infos[i]])):
                print(f"{i}-{j}")

                info = json_dict[meta_infos[i]][j]    
                initial_report, refined_report, assist_infos, initial_answer, refined_answer, logit, confidence_level = self.process(info, medical_field=meta_infos[i])
                answer = info["answer_idx"]
                
                this_json = {'initial_report': initial_report, 'refined_report': refined_report, "assist_infos":assist_infos, 'initial_answer': initial_answer, 'refined_answer': refined_answer, 'answer': answer,'logit': logit}
                self.save_file(this_json, output_dir+f"/{meta_infos[i]}/{j}.json")
                
                results[meta_infos[i]]['initial_answer'].append(initial_answer)
                results[meta_infos[i]]['refined_answer'].append(refined_answer)
                results[meta_infos[i]]['scores'].append(logit)
                results[meta_infos[i]]['targets'].append(answer)

                print(f"Initial Answer: {initial_answer}, Refined Answer: {refined_answer}, Confidence_level: {confidence_level}, Target: {answer}")
            self.save_file(results[meta_infos[i]], output_dir+f"/results_{meta_infos[i]}.json")
        
        self.save_file(results, output_dir + "/results.json")
        
    def save_file(self, file, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(file, f, indent=4, ensure_ascii=False)

