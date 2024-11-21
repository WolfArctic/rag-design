import dspy
import time
# api_base = "https://api.zhizengzeng.com/v1"
# api_key = "sk-zk2f0f4778387b3d6caa738d524269076986721a7d7b7b86"
api_key = "ollama"


# lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, api_base=api_base)
start_time = time.time()
# lm = dspy.LM('ollama_chat/qwen2.5:7b', api_key=api_key, api_base="http://127.0.0.1:11434")
# lm = dspy.LM('ollama_chat/llama3.2:3b', api_key=api_key, api_base="http://127.0.0.1:11434")
lm = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434', api_key='')
# lm = dspy.LM('ollama_chat/deepseek-v2:16b', api_key=api_key, api_base="http://127.0.0.1:11434")
end_time = time.time()
print(f"\n dspy.LM 执行时间: {end_time - start_time:.4f} 秒")

start_time = time.time()
dspy.configure(lm=lm)
end_time = time.time()
print(f"\n config 执行时间: {end_time - start_time:.4f} 秒")

start_time = time.time()
qa = dspy.Predict('question: str -> response: str')
end_time = time.time()
print(f"\n Predict 执行时间: {end_time - start_time:.4f} 秒")

start_time = time.time()
# response = qa(question="what are high memory and low memory on linux?")
response = qa(question="你是哪个公司的模型?叫什么名字? 给我唱首歌.")
end_time = time.time()
print(f"\n qa 执行时间: {end_time - start_time:.4f} 秒")

start_time = time.time()
print("===============")
print(response.response)
end_time = time.time()
print(f"\n print 执行时间: {end_time - start_time:.4f} 秒")
print("===============")
# cot = dspy.ChainOfThought('question -> response')
# cot(question="should curly braces appear on their own line?")
import time
while False:
    lm_test = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm_test)
    qa_test = dspy.Predict('question: str -> response: str')
    response_test = qa(question="你是哪个公司的模型?叫什么名字? 给我唱首歌.")
    print(response_test.response)
    time.sleep(0.1)

dspy.inspect_history(n=1)
