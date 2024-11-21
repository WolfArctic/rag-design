from litellm import completion

response = completion(
    model="ollama/deepseek-v2:16b", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://127.0.0.1:11434"
)
print(response)

response = completion(
    model="ollama_chat/deepseek-v2:16b",
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}],
    api_base="http://127.0.0.1:11434"
)
print(response)
