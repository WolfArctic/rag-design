



# 环境变量

在当前目录下新建.env文件，内容如下：

```bash
# 模型名称
#MODEL_NAME = "deepseek-v2:16b"
MODEL_NAME = "llama3.2:3b"
# 基础URL
BASE_URL = "http://127.0.0.1:11434/v1"
OPENAI_API_KEY = "ollama"

# RAG URL
RAG_URL = "https://127.0.0.1:11434/v1"
# RAG TOKEN
RAG_TOKEN = "ollama"

# 数据文档路径
DATA_DOC_PATH = "./data"
# 分词器路径
TOKENIZER_PATH = "./tokenizer.json"
```

