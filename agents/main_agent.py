from enum import Enum
from .base_agent import *
from .prompt import *
import json 

class MainAgent(BaseAgent):    
    def __init__(self, model_key: str = "qwen-vl-max", temperature: float = 0.3):
            """
            Initialize a doctor agent.

            Args:
                agent_id: Unique identifier for the doctor
                specialty: Doctor's medical specialty
                model_key: LLM model to use
            """
            super().__init__(AgentType.ANALYST, model_key,temperature)

    def initial_judgement(self, info, medical_field = None, max_retries=3):
        question = info["question"]
        options = info["options"]

        system_content = general_prompt["system"].render(field=medical_field)
        user_content = general_prompt["user"].render(question=question, options=options)

        res = self.call_llm(system_content, user_content, max_retries)
        logits = self.get_logits()

        if res is None or logits == []:
            return False, {}
        
        res = self.str2json(res)

        ans = {
            "pred_answer": res["Answer"],
            "logits": logits,
            'answer': info["answer"],
            'confidence':res['conf_each_option']
        }
        
        return True, ans 

    def refined_judgement_debate(self, info, pred_answer, assist_infos, medical_field = None, max_retries=3):
        question = info["question"]

        system_content = refined_debate["system"].render(field=medical_field)
        user_content = refined_debate["user"].render(question=question, initial_judgement = pred_answer, assist_infos = assist_infos)
            
        res = self.call_llm(system_content, user_content, max_retries)

        if res is None:
            return pred_answer, {}
        
        res_list = res.split("\",")
        ans = res_list[0].split(": ")[1].split("\"")[1]
        confidence_level = res_list[1].split(": ")[1].split("\"")[1]
        reasoning = res_list[2].split(": ")[1].split("\n}")[0]

        # refined_answer = ans if confidence_level == "high" else pred_answer
        refined_answer = ans 
        res = {
            "Answer": refined_answer,
            "confidence_level": confidence_level,
            "Reasoning": reasoning
        }
        
        return refined_answer, res

    def extract_better_answer(self,json_response):
        """
        保存JSON响应并提取better_answer字段

        Args:
            json_response: 字符串格式的JSON响应或字典对象
        
        Returns:
            str: better_answer的值
        """

        # 如果输入是字符串，解析为字典
        json_response = json_response[json_response.find("{"):json_response.rfind("}")+1]
        if isinstance(json_response, str):
            try:
                data = json.loads(json_response)
            except:
                print("解析JSON响应时出错")
                return False, 'A', 'Low', None
        else:
            data = json_response

        # 提取better_answer
        better_answer = data.get('final_judgment', {}).get('better_answer')
        confidence_level = data.get('final_judgment', {}).get('confidence_level')

        return True, better_answer, confidence_level, data 

