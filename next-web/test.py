#!/usr/bin/env python
# coding=utf-8

import openai
import os
import requests
import time
import json
from openai import OpenAI

# 从 .env 文件加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 获取环境变量
MODEL_NAME = os.getenv('MODEL_NAME', 'deepseek-v2:16b')  # 默认值为 deepseek-v2:16b
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:11434/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'ollama')

# 设置 OpenAI API 配置
openai.api_base = BASE_URL 
openai.api_key = OPENAI_API_KEY


def chat_completions():
    url=openai.api_base+"/chat/completions"
    api_secret_key = openai.api_key;  # 你的api_secret_key
    headers = {'Content-Type': 'application/json', 'Accept':'application/json',
               'Authorization': "Bearer "+api_secret_key}
    params = {'user':'张三',
              'messages':[{'role':'user', 'content':'Give me five countries with letter A in the third position in the name'}]};
    start_time = time.time()
    r = requests.post(url, json.dumps(params), headers=headers)
    end_time = time.time()
    print(f"Request took {end_time - start_time:.2f} seconds")
    print(r.json())

if __name__ == '__main__':
    client = OpenAI(api_key=OPENAI_API_KEY,base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Give me five countries with letter A in the third position in the name."
            }
        ]
    )

    print(response.choices[0].message.content)

