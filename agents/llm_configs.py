import os 

LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 Official",
        "reasoning": False,
    },
    'kimi-k2':{
        "base_url" : "https://api.moonshot.cn/v1",
        "api_key" : KIMI_API_KEY,
        "model_name" : "kimi-k2-0711-preview"
    },
    'gpt-4o': {
        'api_key': OPENAI_API_KEY,
        'model_name': 'gpt-4o',
    },
    'gpt-4o-mini': {
        'api_key': OPENAI_API_KEY,
        'model_name': 'gpt-4o-mini-2024-07-18',
    },
    "conformal_predictor": {
    }
}