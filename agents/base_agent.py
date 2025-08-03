from openai import OpenAI
from enum import Enum
from typing import Dict, Any, Optional, List
import time
from .llm_configs import LLM_MODELS_SETTINGS
import math
import json 

class AgentType(Enum):
    """Agent type enumeration."""
    ANALYST = "Analyst"
    CONFORMAL_PREDICTOR = "Conformal Predictor"
    Assistant = "Assistant"

class BaseAgent:
    """Base class for all agents."""

    def __init__(self,
        agent_type: AgentType,
        model_key: str = "qwen-vl-max",
        temperature: float = 0.0):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor, Coordinator, or Decision Maker)
            model_key: LLM model to use
        """
        self.agent_type = agent_type
        self.model_key = model_key
        self.completion = None
        self.temperature = temperature
        self.total_tokens = 0
        self.completion_tokens = 0

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        if model_key == 'conformal_predictor':
            print("Creating conformal predictor agent without a model key.")
            return 

        # Set up OpenAI client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        if "gpt" in self.model_key:
            self.client = OpenAI(
                api_key=model_settings["api_key"],
            )
        else:
            self.client = OpenAI(
                api_key=model_settings["api_key"],
                base_url=model_settings["base_url"],
            )
        self.model_name = model_settings["model_name"]

    def str2json(self,response):
        response = response[response.find("{"):response.rfind("}")+1]
        if isinstance(response, str):
            try:
                # Attempt to parse the response as JSON
                response = json.loads(response)
            except json.JSONDecodeError:
                print("Response is not valid JSON, returning as string.")
        else:
            response = response.to_dict() if hasattr(response, 'to_dict') else response
        return response

    def get_logits(self):
        logprobs = self.completion.choices[0].logprobs.content
        logits = self.get_logprobs(logprobs)
        return logits

    def call_llm(self,
                system_message: Dict[str, str],
                user_message: Dict[str, Any],
                max_retries: int = 5,
                return_json = True) -> str:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing question and optional image
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0

        system_message = {'role': 'system', 'content': system_message}
        user_message = {'role': 'user', 'content': user_message}
        while retries < max_retries:

            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    logprobs=True,
                    top_logprobs=5,
                    temperature=self.temperature,
                )
                
                self.completion_tokens += completion.usage.completion_tokens
                self.total_tokens += completion.usage.total_tokens
                
                response = completion.choices[0].message.content
                self.completion = completion
                
                if not return_json:
                    return response
                response = response[response.find("{"):response.rfind("}")+1]
                
                return response
            
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    print(f"LLM API call failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(1)  # Brief pause before retrying

        return None

    def get_logprobs(self, logprobs):

        probs={}
        res=[0,0,0,0,0]
        a_map={'A':0,'B':1,'C':2,'D':3,'E':4}

        for token_info in logprobs:
            token = token_info.token
            top_logprobs = token_info.top_logprobs
            if token in ['A', 'B', 'C', 'D', 'E']:     
                total = 0
                for i in top_logprobs:
                    in_token = i.token
                    if in_token in ['A', 'B', 'C', 'D', 'E']:        
                        i_token = in_token
                        i_logprob = i.logprob
                        probs[i_token] = math.exp(i_logprob)
                        total+=math.exp(i_logprob)
                for i in probs:
                    res[a_map[i]]=round(probs[i]/total,5)
                return res

        return []

