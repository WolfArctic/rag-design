import json
# 系统标准库导入
import os
import time
import traceback
import statistics
from typing import Any, Dict, Generator, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# 第三方库导入
import byzerllm
from byzerllm import ByzerLLM
from loguru import logger
from openai import OpenAI
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from tokenizers import Tokenizer
import pathspec

# 项目内部导入
from autocoder.common import AutoCoderArgs, SourceCode
from doc_filter import DocFilter
from autocoder.rag.document_retriever import LocalDocumentRetriever
from autocoder.rag.relevant_utils import (
    DocRelevance,
    FilterDoc,
    TaskTiming, 
    parse_relevance,
)
from autocoder.rag.token_checker import check_token_limit
from autocoder.rag.token_counter import RemoteTokenCounter, TokenCounter
from autocoder.rag.variable_holder import VariableHolder

load_dotenv()

# 获取环境变量
MODEL_NAME = os.getenv('MODEL_NAME', 'deepseek-v2:16b')  # 默认值为 deepseek-v2:16b
BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:11434/v1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'ollama')

RAG_URL = os.getenv('RAG_URL', 'http://127.0.0.1:11434')
RAG_TOKEN = os.getenv('RAG_TOKEN', 'ollama')


class TokenLimiter:
    def __init__(
        self,
        count_tokens: Callable[[str], int],
        full_text_limit: int,
        segment_limit: int,
        buff_limit: int,
        llm,
        disable_segment_reorder: bool,
    ):
        self.count_tokens = count_tokens
        self.full_text_limit = full_text_limit
        self.segment_limit = segment_limit
        self.buff_limit = buff_limit
        self.llm = llm
        self.first_round_full_docs = []
        self.second_round_extracted_docs = []
        self.sencond_round_time = 0
        self.disable_segment_reorder = disable_segment_reorder

    @byzerllm.prompt()
    def extract_relevance_range_from_docs_with_conversation(
        self, conversations: List[Dict[str, str]], documents: List[str]
    ) -> str:
        """
        根据提供的文档和对话历史提取相关信息范围。

        输入:
        1. 文档内容:
        {% for doc in documents %}
        {{ doc }}
        {% endfor %}

        2. 对话历史:
        {% for msg in conversations %}
        <{{ msg.role }}>: {{ msg.content }}
        {% endfor %}

        任务:
        1. 分析最后一个用户问题及其上下文。
        2. 在文档中找出与问题相关的一个或多个重要信息段。
        3. 对每个相关信息段，确定其起始行号(start_line)和结束行号(end_line)。
        4. 信息段数量不超过4个。

        输出要求:
        1. 返回一个JSON数组，每个元素包含"start_line"和"end_line"。
        2. start_line和end_line必须是整数，表示文档中的行号。
        3. 行号从1开始计数。
        4. 如果没有相关信息，返回空数组[]。

        输出格式:
        严格的JSON数组，不包含其他文字或解释。

        示例:
        1.  文档：
            1 这是这篇动物科普文。
            2 大象是陆地上最大的动物之一。
            3 它们生活在非洲和亚洲。
            问题：大象生活在哪里？
            返回：[{"start_line": 2, "end_line": 3}]

        2.  文档：
            1 地球是太阳系第三行星，
            2 有海洋、沙漠，温度适宜，
            3 是已知唯一有生命的星球。
            4 太阳则是太阳系的唯一恒心。
            问题：地球的特点是什么？
            返回：[{"start_line": 1, "end_line": 3}]

        3.  文档：
            1 苹果富含维生素。
            2 香蕉含有大量钾元素。
            问题：橙子的特点是什么？
            返回：[]
        """

    def limit_tokens(
        self,
        relevant_docs: List[SourceCode],
        conversations: List[Dict[str, str]],
        index_filter_workers: int,
    ) -> List[SourceCode]:
        final_relevant_docs = []
        token_count = 0
        doc_num_count = 0

        reorder_relevant_docs = []

        ## 文档分段（单个文档过大）和重排序逻辑
        ## 1. 背景：在检索过程中，许多文档被切割成多个段落（segments）
        ## 2. 问题：这些segments在召回时因为是按相关分做了排序可能是乱序的，不符合原文顺序，会强化大模型的幻觉。
        ## 3. 目标：重新排序这些segments，确保来自同一文档的segments保持连续且按正确顺序排列。
        ## 4. 实现方案：
        ##    a) 方案一（保留位置）：统一文档的不同segments 根据chunk_index 来置换位置
        ##    b) 方案二（当前实现）：遍历文档，发现某文档的segment A，立即查找该文档的所有其他segments，
        ##       对它们进行排序，并将排序后多个segments插入到当前的segment A 位置中。
        ## TODO:
        ##     1. 未来根据参数决定是否开启重排以及重排的策略
        if not self.disable_segment_reorder:
            num_count = 0
            for doc in relevant_docs:
                num_count += 1
                reorder_relevant_docs.append(doc)
                if "original_doc" in doc.metadata and "chunk_index" in doc.metadata:
                    #original_doc_name = doc.metadata["original_doc"].module_name
                    original_doc_name = doc.metadata["original_doc"]

                    temp_docs = []
                    for temp_doc in relevant_docs[num_count:]:
                        if (
                            "original_doc" in temp_doc.metadata
                            and "chunk_index" in temp_doc.metadata
                        ):
                            if (
                                #temp_doc.metadata["original_doc"].module_name
                                temp_doc.metadata["original_doc"]
                                == original_doc_name
                            ):
                                if temp_doc not in reorder_relevant_docs:
                                    temp_docs.append(temp_doc)

                    temp_docs.sort(key=lambda x: x.metadata["chunk_index"])
                    reorder_relevant_docs.extend(temp_docs)
        else:
            reorder_relevant_docs = relevant_docs

        ## 非窗口分区实现
        for doc in reorder_relevant_docs:
            doc_tokens = self.count_tokens(doc.source_code)
            doc_num_count += 1
            if token_count + doc_tokens <= self.full_text_limit + self.segment_limit:
                final_relevant_docs.append(doc)
                token_count += doc_tokens
            else:
                break

        ## 如果窗口无法放下所有的相关文档，则需要分区
        if len(final_relevant_docs) < len(reorder_relevant_docs):
            ## 先填充full_text分区
            token_count = 0
            new_token_limit = self.full_text_limit
            doc_num_count = 0
            for doc in reorder_relevant_docs:
                doc_tokens = self.count_tokens(doc.source_code)
                doc_num_count += 1
                if token_count + doc_tokens <= new_token_limit:
                    self.first_round_full_docs.append(doc)
                    token_count += doc_tokens
                else:
                    break

            if len(self.first_round_full_docs) > 0:
                remaining_tokens = (
                    self.full_text_limit + self.segment_limit - token_count
                )
            else:
                logger.warning(
                    "Full text area is empty, this is may caused by the single doc is too long"
                )
                remaining_tokens = self.full_text_limit + self.segment_limit

            ## 继续填充segment分区
            sencond_round_start_time = time.time()
            remaining_docs = reorder_relevant_docs[len(self.first_round_full_docs) :]
            logger.info(
                f"first round docs: {len(self.first_round_full_docs)} remaining docs: {len(remaining_docs)} index_filter_workers: {index_filter_workers}"
            )

            with ThreadPoolExecutor(max_workers=index_filter_workers or 5) as executor:
                future_to_doc = {
                    executor.submit(self.process_range_doc, doc, conversations): doc
                    for doc in remaining_docs
                }

                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        result = future.result()
                        if result and remaining_tokens > 0:
                            self.second_round_extracted_docs.append(result)
                            tokens = result.tokens
                            if tokens > 0:
                                remaining_tokens -= tokens
                            else:
                                logger.warning(
                                    f"Token count for doc {doc.module_name} is 0 or negative"
                                )
                    except Exception as exc:
                        logger.error(
                            f"Processing doc {doc.module_name} generated an exception: {exc}"
                        )

            final_relevant_docs = (
                self.first_round_full_docs + self.second_round_extracted_docs
            )

            self.sencond_round_time = time.time() - sencond_round_start_time
            logger.info(
                f"Second round processing time: {self.sencond_round_time:.2f} seconds"
            )

        return final_relevant_docs

    def process_range_doc(
        self, doc: SourceCode, conversations: List[Dict[str, str]], max_retries=3
    ) -> SourceCode:
        for attempt in range(max_retries):
            content = ""
            try:
                source_code_with_line_number = ""
                source_code_lines = doc.source_code.split("\n")
                for idx, line in enumerate(source_code_lines):
                    source_code_with_line_number += f"{idx+1} {line}\n"

                #llm = ByzerLLM()
                #llm.setup_default_model_name(self.llm.default_model_name)
                #llm.skip_nontext_check = True
                llm = OpenAILLM(
                    model_name=MODEL_NAME,
                    api_key=OPENAI_API_KEY, 
                    base_url = BASE_URL)

                '''
                extracted_info = (
                    self.extract_relevance_range_from_docs_with_conversation.options(
                        {"llm_config": {"max_length": 100}}
                    )
                    .with_llm(llm)
                    .run(conversations, [source_code_with_line_number])
                )
                '''
                llm_config = {}
                #query = conversations[-1]["content"]
                extracted_info_prompt = self.extract_relevance_range_from_docs_with_conversation(conversations, [source_code_with_line_number])
                extracted_content = []
                response = llm.client.chat.completions.create(
                    messages=[{"role": "user", "content": extracted_info_prompt}],
                    model=MODEL_NAME
                )

                json_str = response.choices[0].message.content
                content = ""
                try:
                    json_objs = json.loads(json_str)
                except Exception as e:
                    json_objs = []
                if json_objs:
                    for json_obj in json_objs:
                        start_line = json_obj["start_line"] - 1
                        end_line = json_obj["end_line"]
                        chunk = "\n".join(source_code_lines[start_line:end_line])
                        content += chunk + "\n"

                return SourceCode(
                    module_name=doc.module_name,
                    source_code=content.strip(),
                    tokens=self.count_tokens(content),
                    metadata={
                        "original_doc": doc.module_name,
                        "chunk_ranges": json_objs,
                    },
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Error processing doc {doc.module_name}, retrying... (Attempt {attempt + 1}) Error: {str(e)}"
                    )
                else:
                    logger.error(
                        f"Failed to process doc {doc.module_name} after {max_retries} attempts: {str(e)}"
                    )
                    return SourceCode(
                        module_name=doc.module_name, source_code="", tokens=0
                    )


class OpenAILLM:
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model_name = model_name

    def setup_default_model_name(self, model_name: str) -> None:
        self.default_model_name = model_name

    def stream_chat_oai(
        self,
        conversations: List[Dict[str, str]],
        model: Optional[str] = None,
        role_mapping: Optional[Dict[str, str]] = None,
        llm_config: Dict[str, Any] = {},
        delta_mode: bool = False,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        try:
            model = model or self.default_model_name
            
            # Apply role mapping if provided
            if role_mapping:
                mapped_conversations = []
                for conv in conversations:
                    mapped_conv = conv.copy()
                    if conv["role"] in role_mapping:
                        mapped_conv["role"] = role_mapping[conv["role"]]
                    mapped_conversations.append(mapped_conv)
            else:
                mapped_conversations = conversations

            response = self.client.chat.completions.create(
                model=model,
                messages=mapped_conversations,
                stream=True,
                **llm_config
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    if delta_mode:
                        yield (chunk.choices[0].delta.content, {})
                    else:
                        yield chunk.choices[0].delta.content, {}
                        
        except Exception as e:
            logger.error(f"Error in stream_chat_oai: {str(e)}")
            traceback.print_exc()
            if delta_mode:
                yield ("Error occurred during API call", {})
            else:
                yield "Error occurred during API call", {}



class LongContextRAG:
    def __init__(
        self,
        llm: OpenAILLM,
        args: AutoCoderArgs,
        path: str,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.args = args

        # self.index_model = OpenAILLM(
        #     model_name=MODEL_NAME,
        #     api_key=OPENAI_API_KEY, 
        #     base_url = BASE_URL)

        self.path = path
        self.relevant_score = self.args.rag_doc_filter_relevance or 5

        self.full_text_ratio = args.full_text_ratio
        self.segment_ratio = args.segment_ratio
        self.buff_ratio = 1 - self.full_text_ratio - self.segment_ratio

        if self.buff_ratio < 0:
            raise ValueError(
                "The sum of full_text_ratio and segment_ratio must be less than or equal to 1.0"
            )

        self.full_text_limit = int(args.rag_context_window_limit * self.full_text_ratio)
        self.segment_limit = int(args.rag_context_window_limit * self.segment_ratio)
        self.buff_limit = int(args.rag_context_window_limit * self.buff_ratio)

        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.on_ray = False

        if self.tokenizer_path:
            VariableHolder.TOKENIZER_PATH = self.tokenizer_path
            VariableHolder.TOKENIZER_MODEL = Tokenizer.from_file(self.tokenizer_path)
            self.tokenizer = TokenCounter(self.tokenizer_path)
        else:
            if llm.is_model_exist("deepseek_tokenizer"):
                tokenizer_llm = ByzerLLM()
                tokenizer_llm.setup_default_model_name("deepseek_tokenizer")
                self.tokenizer = RemoteTokenCounter(tokenizer_llm)

        self.required_exts = (
            [ext.strip() for ext in self.args.required_exts.split(",")]
            if self.args.required_exts
            else []
        )

        # if open monitor mode
        self.monitor_mode = self.args.monitor_mode or False
        self.enable_hybrid_index = self.args.enable_hybrid_index
        logger.info(f"Monitor mode: {self.monitor_mode}")

        if args.rag_url and args.rag_url.startswith("http://"):
            if not args.rag_token:
                raise ValueError(
                    "You are in client mode, please provide the RAG token. e.g. rag_token: your_token_here"
                )
            self.client = OpenAI(api_key=args.rag_token, base_url=args.rag_url)
        else:
            self.client = None
            # if not pure client mode, then the path should be provided
            if (
                not self.path
                and args.rag_url
                and not args.rag_url.startswith("http://")
            ):
                self.path = args.rag_url

            if not self.path:
                raise ValueError(
                    "Please provide the path to the documents in the local file system."
                )
        
        self.ignore_spec = self._load_ignore_file()

        self.token_limit = self.args.rag_context_window_limit or 120000
        retriever_class = self._get_document_retriever_class()
        self.document_retriever = retriever_class(
            self.path,
            self.ignore_spec,
            self.required_exts,
            self.on_ray,
            self.monitor_mode,
            ## 确保全文区至少能放下一个文件
            single_file_token_limit=self.full_text_limit - 100,
            disable_auto_window=self.args.disable_auto_window,            
            enable_hybrid_index=self.args.enable_hybrid_index,
            extra_params=self.args
        )

        self.doc_filter = DocFilter(
            api_key=OPENAI_API_KEY, args=self.args, base_url=BASE_URL, model_name=MODEL_NAME
        )

        doc_num = 0
        token_num = 0
        token_counts = []
        for doc in self._retrieve_documents():
            doc_num += 1
            doc_tokens = doc.tokens
            token_num += doc_tokens
            token_counts.append(doc_tokens)

        avg_tokens = statistics.mean(token_counts) if token_counts else 0
        median_tokens = statistics.median(token_counts) if token_counts else 0

        logger.info(
            "RAG Configuration:\n"
            f"  Total docs:        {doc_num}\n"
            f"  Total tokens:      {token_num}\n"
            f"  Tokenizer path:    {self.tokenizer_path}\n"
            f"  Relevant score:    {self.relevant_score}\n"
            f"  Token limit:       {self.token_limit}\n"
            f"  Full text limit:   {self.full_text_limit}\n"
            f"  Segment limit:     {self.segment_limit}\n"
            f"  Buff limit:        {self.buff_limit}\n"
            f"  Max doc tokens:    {max(token_counts) if token_counts else 0}\n"
            f"  Min doc tokens:    {min(token_counts) if token_counts else 0}\n"
            f"  Avg doc tokens:    {avg_tokens:.2f}\n"
            f"  Median doc tokens: {median_tokens:.2f}\n"
        )

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return -1
        return self.tokenizer.count_tokens(text)

    @byzerllm.prompt()
    def extract_relevance_info_from_docs_with_conversation(
        self, conversations: List[Dict[str, str]], documents: List[str]
    ) -> str:
        """
        使用以下文档和对话历史来提取相关信息。

        文档：
        {% for doc in documents %}
        {{ doc }}
        {% endfor %}

        对话历史：
        {% for msg in conversations %}
        <{{ msg.role }}>: {{ msg.content }}
        {% endfor %}

        请根据提供的文档内容、用户对话历史以及最后一个问题，提取并总结文档中与问题相关的重要信息。
        如果文档中没有相关信息，请回复"该文档中没有与问题相关的信息"。
        提取的信息尽量保持和原文中的一样，并且只输出这些信息。
        """
        #使用以下文档来回答问题。如果文档中没有相关信息，请说"我没有足够的信息来回答这个问题"。

    @byzerllm.prompt()
    def _answer_question(
        self, query: str, relevant_docs: List[str]
    ) -> Generator[str, None, None]:
        """
        使用以下文档来回答问题。如果文档中没有相关信息，请说"文档中没有直接信息，我的推理结果如下:"并给出基于所有文档合并后的推理结果。

        文档：
        {% for doc in relevant_docs %}
        {{ doc }}
        {% endfor %}

        问题：{{ query }}

        回答：
        """

    def _get_document_retriever_class(self):
        """Get the document retriever class based on configuration."""
        # Default to LocalDocumentRetriever if not specified
        return LocalDocumentRetriever
    
    def _load_ignore_file(self):
        serveignore_path = os.path.join(self.path, ".serveignore")
        gitignore_path = os.path.join(self.path, ".gitignore")

        if os.path.exists(serveignore_path):
            with open(serveignore_path, "r") as ignore_file:
                return pathspec.PathSpec.from_lines("gitwildmatch", ignore_file)
        elif os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as ignore_file:
                return pathspec.PathSpec.from_lines("gitwildmatch", ignore_file)
        return None

    def _retrieve_documents(self,options:Optional[Dict[str,Any]]=None) -> Generator[SourceCode, None, None]:
        return self.document_retriever.retrieve_documents(options=options)

    def build(self):
        pass

    def search(self, query: str) -> List[SourceCode]:
        target_query = query
        only_contexts = False
        if self.args.enable_rag_search and isinstance(self.args.enable_rag_search, str):
            target_query = self.args.enable_rag_search
        elif self.args.enable_rag_context and isinstance(
            self.args.enable_rag_context, str
        ):
            target_query = self.args.enable_rag_context
            only_contexts = True
        elif self.args.enable_rag_context:
            only_contexts = True

        logger.info("Search from RAG.....")
        logger.info(f"Query: {target_query[0:100]}... only_contexts: {only_contexts}")

        if self.client:
            new_query = json.dumps(
                {"query": target_query, "only_contexts": only_contexts},
                ensure_ascii=False,
            )
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": new_query}],
                model=self.args.model,
            )
            v = response.choices[0].message.content
            if not only_contexts:
                return [SourceCode(module_name=f"RAG:{target_query}", source_code=v)]

            json_lines = [json.loads(line) for line in v.split("\n") if line.strip()]
            return [SourceCode.model_validate(json_line) for json_line in json_lines]
        else:
            if only_contexts:
                return [
                    doc.source_code
                    for doc in self._filter_docs(
                        [{"role": "user", "content": target_query}]
                    )
                ]
            else:
                v, contexts = self.stream_chat_oai(
                    conversations=[{"role": "user", "content": target_query}]
                )
                url = ",".join(contexts)
                return [SourceCode(module_name=f"RAG:{url}", source_code="".join(v))]

    def _filter_docs(self, conversations: List[Dict[str, str]]) -> List[FilterDoc]:
        query = conversations[-1]["content"]
        documents = self._retrieve_documents(options={"query":query})
        return self.doc_filter.filter_docs(
            conversations=conversations, documents=documents
        )

    def stream_chat_oai(
        self,
        conversations,
        model: Optional[str] = None,
        role_mapping=None,
        llm_config: Dict[str, Any] = {},
    ):
        try:
            return self._stream_chat_oai(
                conversations,
                model=model,
                role_mapping=role_mapping,
                llm_config=llm_config,
            )
        except Exception as e:
            logger.error(f"Error in stream_chat_oai: {str(e)}")
            traceback.print_exc()
            return ["出现错误，请稍后再试。"], []

    def _stream_chat_oai(
        self,
        conversations,
        model: Optional[str] = None,
        role_mapping=None,
        llm_config: Dict[str, Any] = {},
    ):
        
        if self.client:
            model = model or self.args.model
            response = self.client.chat.completions.create(
                model=model,
                messages=conversations,
                stream=True,
            )

            def response_generator():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return response_generator(), []
        else:
            query = conversations[-1]["content"]
            context = []

            if (
                "使用四到五个字直接返回这句话的简要主题，不要解释、不要标点、不要语气词、不要多余文本，不要加粗，如果没有主题"
                in query
                or "简要总结一下对话内容，用作后续的上下文提示 prompt，控制在 200 字以内"
                in query
            ):
                chunks = self.llm.stream_chat_oai(
                    conversations=conversations,
                    model=model,
                    role_mapping=role_mapping,
                    llm_config=llm_config,
                    delta_mode=True,
                )
                return (chunk[0] for chunk in chunks), context

            only_contexts = False
            try:
                v = json.loads(query)
                if "only_contexts" in v:
                    query = v["query"]
                    only_contexts = v["only_contexts"]
            except json.JSONDecodeError:
                pass

            #logger.info(f"Query: {query} only_contexts: {only_contexts}")
            start_time = time.time()

            relevant_docs: List[FilterDoc] = self._filter_docs(conversations)
            filter_time = time.time() - start_time

            # Filter relevant_docs to only include those with is_relevant=True
            highly_relevant_docs = [
                doc for doc in relevant_docs if doc.relevance.is_relevant
            ]

            if highly_relevant_docs:
                relevant_docs = highly_relevant_docs
                logger.info(f"Found {len(relevant_docs)} highly relevant documents")
            else:
                if relevant_docs:
                    prefix_chunk = FilterDoc(
                        source_code=SourceCode(
                            module_name="特殊说明",
                            source_code="没有找到特别相关的内容，下面的内容是一些不是很相关的文档。在根据后续文档回答问题前，你需要和用户先提前说一下。",
                        ),
                        relevance=DocRelevance(False, 0),
                    )
                    relevant_docs.insert(0, prefix_chunk)
                    logger.info(
                        "No highly relevant documents found. Added a prefix chunk to indicate this."
                    )

            logger.info(
                f"Filter time: {filter_time:.2f} seconds with {len(relevant_docs)} docs"
            )

            if only_contexts:
                return (
                    doc.source_code.model_dump_json() + "\n" for doc in relevant_docs
                ), []

            if not relevant_docs:
                return ["没有找到相关的文档来回答这个问题。"], []

            context = [doc.source_code.module_name for doc in relevant_docs]

            # 将 FilterDoc 转化为 SourceCode 方便后续的逻辑继续做处理
            relevant_docs = [doc.source_code for doc in relevant_docs]

            # Create a table for the query information
            query_table = Table(title="Query Information", show_header=False)
            query_table.add_row("Query", query)
            query_table.add_row("Relevant docs", str(len(relevant_docs)))

            # Add relevant docs information
            relevant_docs_info = []
            for doc in relevant_docs:
                info = f"- {doc.module_name.replace(self.path,'',1)}"
                if "original_docs" in doc.metadata:
                    original_docs = ", ".join(
                        [
                            doc.replace(self.path, "", 1)
                            for doc in doc.metadata["original_docs"]
                        ]
                    )
                    info += f" (Original docs: {original_docs})"
                relevant_docs_info.append(info)

            relevant_docs_info = "\n".join(relevant_docs_info)
            query_table.add_row("Relevant docs list", relevant_docs_info)

            first_round_full_docs = []
            second_round_extracted_docs = []
            sencond_round_time = 0

            if self.tokenizer is not None:
                token_limiter = TokenLimiter(
                    count_tokens=self.count_tokens,
                    full_text_limit=self.full_text_limit,
                    segment_limit=self.segment_limit,
                    buff_limit=self.buff_limit,
                    llm=self.llm,
                    disable_segment_reorder=self.args.disable_segment_reorder,
                )
                final_relevant_docs = token_limiter.limit_tokens(
                    relevant_docs=relevant_docs,
                    conversations=conversations,
                    index_filter_workers=self.args.index_filter_workers or 5,
                )
                first_round_full_docs = token_limiter.first_round_full_docs
                second_round_extracted_docs = token_limiter.second_round_extracted_docs
                sencond_round_time = token_limiter.sencond_round_time

                relevant_docs = final_relevant_docs
            else:
                relevant_docs = relevant_docs[: self.args.index_filter_file_num]

            #logger.info(f"Finally send to model: {len(relevant_docs)}")

            query_table.add_row("Only contexts", str(only_contexts))
            query_table.add_row("Filter time", f"{filter_time:.2f} seconds")
            query_table.add_row("Final relevant docs", str(len(relevant_docs)))
            query_table.add_row(
                "first_round_full_docs", str(len(first_round_full_docs))
            )
            query_table.add_row(
                "second_round_extracted_docs", str(len(second_round_extracted_docs))
            )
            query_table.add_row(
                "Second round time", f"{sencond_round_time:.2f} seconds"
            )

            # Add relevant docs information
            final_relevant_docs_info = []
            for doc in relevant_docs:
                info = f"- {doc.module_name.replace(self.path,'',1)}"
                if "original_docs" in doc.metadata:
                    original_docs = ", ".join(
                        [
                            doc.replace(self.path, "", 1)
                            for doc in doc.metadata["original_docs"]
                        ]
                    )
                    info += f" (Original docs: {original_docs})"
                if "chunk_ranges" in doc.metadata:
                    chunk_ranges = json.dumps(
                        doc.metadata["chunk_ranges"], ensure_ascii=False
                    )
                    info += f" (Chunk ranges: {chunk_ranges})"
                final_relevant_docs_info.append(info)

            final_relevant_docs_info = "\n".join(final_relevant_docs_info)
            query_table.add_row("Final Relevant docs list", final_relevant_docs_info)

            # Create a panel to contain the table
            panel = Panel(
                query_table,
                title="RAG Search Results",
                expand=False,
            )

            request_tokens = sum([doc.tokens for doc in relevant_docs])
            target_model = model or self.llm.default_model_name
            logger.info(
                f"Start to send to model {target_model} with {request_tokens} tokens"
            )

            new_conversations = conversations[:-1] + [
                {
                    "role": "user",
                    "content": self._answer_question.prompt(
                        query=query,
                        relevant_docs=[doc.source_code for doc in relevant_docs],
                    ),
                }
            ]

            chunks = self.llm.stream_chat_oai(
                conversations=new_conversations,
                model=model,
                role_mapping=role_mapping,
                llm_config=llm_config,
                delta_mode=True,
            )

            return (chunk[0] for chunk in chunks), context

def main():
    """
    主函数 - 用于测试 LongContextRAG 的功能
    """
    import byzerllm
    from autocoder.common import AutoCoderArgs
    from loguru import logger
    import os

    # 配置日志
    logger.add("rag_test.log", rotation="500 MB")

    try:
        # 初始化 LLM 模型
        llm = byzerllm.ByzerLLM()
        llm.setup_default_model_name("deepseek-chat")

        # 配置 RAG 参数
        args = AutoCoderArgs()
        args.rag_context_window_limit = 1200000  # 上下文窗口大小
        args.rag_doc_filter_relevance = 5  # 文档相关性过滤阈值
        args.full_text_ratio = 0.3  # 全文匹配比例
        args.segment_ratio = 0.5  # 文本片段比例
        
        # 获取当前目录作为测试路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化 RAG 系统
        rag = LongContextRAG(
            llm=llm,
            args=args,
            path=current_dir,
        )

        # 测试查询列表
        test_queries = [
            "解释一下这个项目中LongContextRAG类的主要功能是什么？",
            "这个项目中如何处理token限制？",
            "项目中的文档过滤机制是如何工作的？"
        ]

        logger.info("Starting RAG system tests")
        
        # 执行测试查询
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Processing test query {i}: {query}")
            
            try:
                # 获取查询响应
                response_generator, contexts = rag.stream_chat_oai(
                    conversations=[{"role": "user", "content": query}]
                )
                
                # 处理响应结果
                response_str = "Response:\n"
                for chunk in response_generator:
                    response_str += chunk
                print(response_str)
                
                # 输出相关文档
                if contexts:
                    logger.info("Found relevant documents:")
                    for context in contexts:
                        logger.info(f"- {context}")
                
            except Exception as e:
                logger.error(f"Failed to process query {i}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise
if __name__ == "__main__":
    main()
