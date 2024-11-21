import os
import shutil
import hashlib
import argparse
import gradio as gr

from vllm import LLM, SamplingParams
from passage_retrieval import Retriever
from generate_passage_embeddings import selfrag_embeddings

from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, UnstructuredWordDocumentLoader,UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
import json

EMBEDDINGS_DIR = './chroma/'
KNOWLEDGE_DIR = './chroma/knowledge/'

def get_md5(input_string):
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()

    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))

    # 获取十六进制的哈希值
    return hash_md5.hexdigest()

def get_embeddings(collection, passages):

    # 创建模型和tokenizer
    model_name = "facebook/contriever-msmarco"

    collection_dir = os.path.join(EMBEDDINGS_DIR, collection)
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)
    print('collection_dir:',collection_dir)

    # 已经向量化过的标志文件
    index_file_path = collection_dir + '/.index'
    if os.path.exists(index_file_path):
        print('已经向量化过')
        return
    

    # 创建参数
    args = argparse.Namespace(
        passages=passages,
        output_dir=collection_dir,
        prefix="passages",
        shard_id=0,
        num_shards=4,
        model_name_or_path=model_name,
        per_gpu_batch_size=512,
        passage_maxlength=512,
        no_fp16=False,
        no_title=False,
        lowercase=False,
        normalize_text=False
    )
    selfrag_embeddings(args)
    # 在目录collection_dir下创建一个.index文件
    if not os.path.exists(index_file_path):
        with open(index_file_path, 'w') as f:
            f.write('')

class MyCustomLoader(BaseLoader):
    # 文件类型和对应的加载器
    file_type = {
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {})
    }

    # 初始化方法  将加载的文件进行切分
    def __init__(self, file_path: str):
        # loader_class, params = self.file_type[detect_filetype(file_path)]
        # print('loader_class:',loader_class)
        # print('params:',params)
        # self.loader: BaseLoader = loader_class(file_path, **params)
        # print('self.loader:',self.loader)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    
    def convert_to_jsonl(self, file_path: str):
        file_type = detect_filetype(file_path)
        if file_type == FileType.PDF:
            loader_class, params = self.file_type[file_type]
            loader = loader_class(file_path, **params)
            data = loader.load()
            
            jsonl_file_path = file_path.replace('.pdf', '.jsonl')
            file_name = os.path.basename(file_path)
            item_title = file_name
            with open(jsonl_file_path, 'w') as f:
                for doc in data:
                    page_id = str(doc.metadata['page'])
                    page_content = doc.page_content
                    # 将Document对象转换为字典以便JSON序列化
                    formatted_item = {"id": page_id, "title": item_title, "section": "", "text": page_content}
                    f.write(json.dumps(formatted_item) + '\n')
            return jsonl_file_path
        else:
            return file_path

    def lazy_load(self):
        # 懒惰切分加载
        return self.loader.load_and_split(self.text_splitter)

    def load(self):
        # 加载
        return self.lazy_load()


class MyKnowledge:
    __retrievers = {}
    __embeddings = None

    def __init__(self):
        pass
        
        

    def upload_knowledge(self, temp_file):
        if temp_file is None:
            return gr.update(choices=self.load_knowledge())
        file_name = os.path.basename(temp_file)
        file_path = os.path.join(KNOWLEDGE_DIR, file_name)
        if not os.path.exists(file_path):
            # 如果文件不存在,那么就创建目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 将temp_file复制到知识库目录下file_path
            shutil.copy(temp_file, file_path)
        return None, gr.update(choices=self.load_knowledge())

    def load_knowledge(self):
        # exist_ok=True目标目录已存在的情况下不会抛出异常。
        # 这意味着如果目录已经存在，os.makedirs不会做任何事情，也不会报错
        os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)

        # 知识库默认为空
        collections = [None]
        print('os.listdir(KNOWLEDGE_DIR):',os.listdir(KNOWLEDGE_DIR))

        for file in os.listdir(KNOWLEDGE_DIR):
            # 得到知识库的路径
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            print('file_path:', file_path)
            # 将文件转换为jsonl格式
            loader = MyCustomLoader(file_path)
            file_path_jsonl = loader.convert_to_jsonl(file_path)
            # 如果文件是pdf格式,则跳过
            if file.endswith('.pdf'):
                continue
            # 将知识库进行添加
            collections.append(file)

            # 知识库文件名进行md5编码,对某一个知识库进行唯一标识
            # collection_name1
            # collection_name2
            collection_name = get_md5(file)
            print('collection_name:',collection_name)

            if collection_name in self.__retrievers:
                continue
            
            get_embeddings(collection_name, file_path_jsonl)
            self.__retrievers[collection_name] = Retriever({})

        # 更新并返回知识库列表
        for file in os.listdir(KNOWLEDGE_DIR):
            if file.endswith('.jsonl'):
                if file not in collections:
                    collections.append(file)
        return collections

    def get_retriever(self, collection):
        collection_name = get_md5(collection)
        print('知识库名字md5:',collection_name)
        if collection_name not in self.__retrievers:
            print('不存在这个 retriver and self.__retrievers:',self.__retrievers)
            return None
        else:
            return self.__retrievers[collection_name]

class MyLLM(MyKnowledge):

    def __init__(self):
        # 初始化 prompts
        self.prompts = []
        self.retriever = None
        self.model = None
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
        self.collection_status = {}
    
    def set_collection_status(self, collection, status):
        self.collection_status[collection] = status

    

    def get_model(self, collection, llm_model=None, retriever_model_name_or_path=None, max_length=256, temperature=1):
        
        if collection:
            self.retriever = self.get_retriever(collection)
            if collection not in self.collection_status:
                self.set_collection_status(collection, False)
            print("retriever = ", self.retriever)
       
        # 只是保留3个聊天记录
        # self.history_message = self.history_message[-3:]

        # 初始化 模型
        if self.model is None:
            self.model = LLM(llm_model, download_dir="/gscratch/h2lab/akari/model_cache", dtype="half", max_model_len=1744)
        chain = self.model

        if self.retriever is not None:
            passages = os.path.join(KNOWLEDGE_DIR, collection)
            passages_embeddings = os.path.join(EMBEDDINGS_DIR, get_md5(collection)) + "/*"
            n_docs = 5
            save_or_load_index = False
            if self.collection_status[collection] is False:
                self.retriever.setup_retriever_demo(retriever_model_name_or_path, passages, passages_embeddings, n_docs, save_or_load_index)
                self.set_collection_status(collection, True)
        return chain
    

    def format_prompt(self, input, paragraph=None):
        prompt = "You are Monkey Master. \
            If you're unsure of the answer, please respond with 'I don't know'."
        prompt += "### Instruction:\n{0}\n\n### Response:\n".format(input)
        if paragraph is not None:
            prompt += "Please ensure your answer is based solely on the provided retrieval."
            prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
            prompt += "\n\n"
        return prompt
    
        # prompt = "Your name is Monkey Master. \
        #     If you don't know the answer, just say 'I don't know'."
        # prompt += "### Instruction:\n{0}\n\n### Response:\n".format(input)
        # if paragraph is not None:
        #     prompt += "Your answer only include the information in the retrieval."
        #     prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
        #     prompt += "\n\n"
        # return prompt


    def clear_history(self):
        self.prompts = []
        

    def get_history_message(self):
        return self.prompts

    def stream(self, query, collection, llm_model=None, retriever_model_name_or_path=None, max_length=256, temperature=1):
        model = self.get_model(collection, llm_model, retriever_model_name_or_path, max_length, temperature)
        if self.retriever is not None:
            retrieved_documents = self.retriever.search_document_demo(query, 3)
            self.prompts = [self.format_prompt(query, doc["title"] +"\n"+ doc["text"]) for doc in retrieved_documents]
        else:
            self.prompts = [self.format_prompt(query)]
        result = model.generate(self.prompts, self.sampling_params)
        # result = model.chat(self.prompts, self.sampling_params)
        return result
    

if __name__ == "__main__":
    llm = MyLLM()
    print(llm.stream("What is the capital of France?", "selfrag/selfrag_llama2_7b"))
