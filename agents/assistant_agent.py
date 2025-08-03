import json
from .prompt import * 
from enum import Enum
from .base_agent import *
from .MedRAG.src.utils import RetrievalSystem
from .conformal_agent import ConformalAgent

class AssistAgent(BaseAgent):    
    def __init__(self, model_key: str = "qwen-vl-max", temperature: float = 0.0):
            """
            Initialize a doctor agent.

            Args:
                agent_id: Unique identifier for the doctor
                specialty: Doctor's medical specialty
                model_key: LLM model to use
            """
            super().__init__(AgentType.Assistant, model_key, temperature)
            # self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
            self.describe = "Assist Agent for medical question answering and analysis."
            self.retrieval_system = RetrievalSystem(retriever_name="MedCPT",db_dir='')
            self.conformal_agent = ConformalAgent(
                model_key="conformal_predictor",
                alpha=0.2
            )
    
    def assist_info_debate(self, info, max_retries=3):
        question = info["question"]
        options = info['options']
        
        is_break = False
        round = 0
        while(not is_break and round < 3):
            is_break = True
            round += 1
            # 首先进行RAG全问题分析
            rag_analysis_results = self.assist_info_rag_whole_question(info, setup="a medical expert", max_retries=max_retries)
            
            # 将RAG分析结果序列化成字符串
            if isinstance(rag_analysis_results, list):
                rag_analysis = "\n\n".join([
                    f"**Analysis for Query: {item['question']}**\n{item['analysis']}" 
                    for item in rag_analysis_results if isinstance(item, dict)
                ])
            else:
                rag_analysis = str(rag_analysis_results)
            
            # 格式化选项
            options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            
            system_content = assist_debate_prompt["system"].render()
            user_content = assist_debate_prompt["user"].render(
                question=question, 
                options=options_str,
                rag_analysis=rag_analysis
            )
            res = self.call_llm(system_content, user_content, max_retries)
            res = self.str2json(res)
            logits = self.get_logits()
            pset = self.conformal_agent.conformal_prediction(logits)
            pset_infos = self.extract_pset(pset, info['options'])
            if len(pset_infos) > 1:
                print("------------ RAG analysis is not confident enough, retrying... ------------")
                is_break = False
                
        return res,rag_analysis

    def assist_info_rag_whole_question(self, info, setup="a medical expert", max_retries=3):
        """
        针对整个问题的RAG分析
        
        Args:
            info (dict): 包含问题信息的字典
            setup (str): 身份设定
            max_retries (int): 最大重试次数
        
        Returns:
            str: 分析结果
        """
        question = info["question"]
        options = info["options"]
        
        # 格式化选项为字符串
        options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        return self._execute_rag_analysis(
            prompt_template=rag_get_question_whole_analysis_prompt,
            analysis_template=rag_whole_analysis_prompt,
            question=question,
            options=options_str,
            setup=setup,
            mode="whole_question",
            max_retries=max_retries
        )

    def _execute_rag_analysis(self, prompt_template, analysis_template, question, mode, max_retries, **kwargs):
        """
        执行RAG分析的通用函数
        
        Args:
            prompt_template: 生成查询问题的prompt模板
            analysis_template: 分析的prompt模板
            question: 主问题
            mode: 分析模式 ("single_option" 或 "whole_question")
            max_retries: 最大重试次数
            **kwargs: 其他参数
        
        Returns:
            list: 分析结果列表
        """
        try:
            # 根据模式准备prompt参数
            
            system_content = prompt_template["system"].render(setup=kwargs.get('setup', 'a medical expert'))
            user_content = prompt_template["user"].render(
                main_question=question,
                options=kwargs.get('options', ''),
                setup=kwargs.get('setup', 'a medical expert')
            )
            
            # 生成RAG查询问题
            query_response = self.call_llm(system_content, user_content, max_retries)
            query_json = self.str2json(query_response)
            
            if not query_json:
                return "生成查询问题失败"
            
            generated_questions = query_json.get("questions", [])
            
            if isinstance(generated_questions, list) and len(generated_questions) > 0:
                retrieved_data = []
                for q in generated_questions:
                    context = self.retrieval_system.retrieve(q, k=3)
                    retrieved_data = retrieved_data + context[0]
                
                formatted_context = "\n\n".join([f"Document [{i+1}]:\n{doc}" for i, doc in enumerate(retrieved_data)])
                
                try:
                    analysis_system = analysis_template["system"].render(setup=kwargs.get('setup', 'a medical expert'))
                    analysis_user = analysis_template["user"].render(
                        question=question,
                        options=kwargs.get('options', ''),
                        formatted_context=formatted_context
                    )
                    
                    analysis_results = self.call_llm(analysis_system, analysis_user, max_retries,return_json=False)
                    
                except Exception as e:
                    print(f"Error generating analysis for question : {e}")
                    analysis_results = "Failed to generate analysis due to an error."
                    
                return analysis_results
            else:
                return "未能生成有效的查询问题"
            
        except Exception as e:
            print(f"RAG分析出错: {e}")
            return "RAG分析失败"

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

