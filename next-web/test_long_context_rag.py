import tempfile
import os
import sys
import json
from long_context_rag import LongContextRAG, OpenAILLM


# 从 .env 文件加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 获取环境变量
MODEL_NAME = os.getenv('MODEL_NAME', 'deepseek-v2:16b')  # 默认值为 deepseek-v2:16b
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:11434/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'ollama')

RAG_URL = os.getenv('RAG_URL', 'http://127.0.0.1:11434')
RAG_TOKEN = os.getenv('RAG_TOKEN', 'ollama')

DATA_DOC_PATH = os.getenv('DATA_DOC_PATH', './data')
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', './tokenizer.json')


class Arguments:
    def __init__(self):
        self.collection = None
        self.disable_inference_enhance = None
        self.inference_deep_thought = None
        self.path = None
        self.rag_context_window_limit = 4096
        self.rag_doc_filter_relevance = 5
        self.full_text_ratio = 0.6 
        self.segment_ratio = 0.4
        self.index_model = None
        self.required_exts = ".pdf"
        self.base_dir = None
        self.collections = None
        self.monitor_mode = False
        self.model = MODEL_NAME
        self.enable_hybrid_index = False
        self.rag_url = RAG_URL
        self.rag_token = RAG_TOKEN
        self.disable_auto_window = False
        self.hybrid_index_max_output_tokens = 4096
        self.api_key = OPENAI_API_KEY
        self.index_filter_workers = 5
        self.tokenizer_path = None
        self.default_model_name = MODEL_NAME
        self.disable_segment_reorder = False

    def verify_stream_chat_output(self, output):
        """Verify the output from stream_chat_oai"""
        # Check if output is a generator
        assert hasattr(output, '__iter__'), "Output should be iterable"
        
        # Collect all responses
        responses = list(output)
        
        # Verify responses are not empty
        assert len(responses) > 0, "No responses received from stream chat"
        
        # Verify each response has expected structure
        for response in responses:
            assert isinstance(response, dict), "Each response should be a dictionary"
            assert 'choices' in response, "Response should contain 'choices'"
            assert len(response['choices']) > 0, "Response should have at least one choice"
            assert 'delta' in response['choices'][0], "Choice should contain 'delta'"
            assert 'content' in response['choices'][0]['delta'], "Delta should contain 'content'"


def filter_query_table(contexts):
    """
    过滤掉 contexts 中的 query table 信息
    Args:
        contexts: 原始的 contexts 列表
    Returns:
        filtered_contexts: 过滤后的 contexts 列表，只包含文件路径
    """
    filtered_contexts = []
    skip_lines = False
    
    for context in contexts:
        # 检查是否是 query table 的开始
        if "RAG Search Results" in context or "Query Information" in context:
            skip_lines = True
            continue
            
        # 检查是否是 query table 的结束
        if skip_lines and "╰──" in context:
            skip_lines = False
            continue
            
        # 如果不在 query table 区域内且是文件路径，则保留
        if not skip_lines and context.strip().startswith('-'):
            # 提取文件路径（去掉前面的 "- " 符号）
            file_path = context.strip()[2:]
            filtered_contexts.append(file_path)
            
    return filtered_contexts


def main(input_data):
    # 初始化参数
    args = Arguments()
    args.path = DATA_DOC_PATH
    args.tokenizer_path = TOKENIZER_PATH
    args.index_model = MODEL_NAME
    
    # 创建 LLM 实例
    llm = OpenAILLM(
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )
    llm.setup_default_model_name(MODEL_NAME)

    # 创建 RAG 实例并执行查询
    rag = LongContextRAG(
        args=args,
        path=args.path, 
        tokenizer_path=args.tokenizer_path,
        llm=llm
    )
    
    response_generator, contexts = rag.stream_chat_oai(
        conversations=[{"role": "user", "content": input_data}]
    )

    # 处理响应
    response_str = ""
    for chunk in response_generator:
        response_str += chunk
    print(json.dumps({"Response": response_str}))
    # trans_text = response_str.encode('utf-8').decode('unicode-escape')
    # print(response_str)
    
    return contexts

if __name__ == '__main__':
    try:
        # 读取输入参数
        input_data = sys.stdin.read()
        # input_data = '{"query": "你是哪个公司的模型?"}'
        # input_data = '{"query": "什么是vLLM?"}'
        params = json.loads(input_data) if input_data else {}

        # 执行主函数
        result = main(input_data)

        # 输出结果
        '''
        print(json.dumps({
            "status": "success",
            "result": result
        }))
        sys.stdout.flush()
        '''
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))
        sys.stderr.flush()
