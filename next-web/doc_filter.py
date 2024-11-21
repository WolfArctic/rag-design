import time
from typing import List, Dict, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from openai import OpenAI

from autocoder.rag.relevant_utils import (
    parse_relevance,
    FilterDoc,
    TaskTiming,
)
from autocoder.common import SourceCode, AutoCoderArgs
from autocoder.rag.rag_config import RagConfigManager, RagConfig

class OpenAIWrapper:
    def __init__(self, api_key: str, base_url: str = None, model_name: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1"
        )
        self.default_model = model_name
    
    def setup_default_model_name(self, model_name: str):
        self.default_model = model_name
    
    def check_relevance(self, conversations: List[Dict[str, str]], documents: List[str], filter_config: Optional[str] = None) -> str:
        # 构建 prompt
        prompt = "使用以下文档和对话历史来回答问题。如果文档中没有相关信息，请说\"没有足够的信息来回答这个问题\"。\n\n"
        # 添加文档
        prompt += "文档：\n"
        for doc in documents:
            prompt += f"{doc}\n"
        
        # 添加对话历史
        prompt += "\n对话历史：\n"
        for msg in conversations:
            prompt += f"<{msg['role']}>: {msg['content']}\n"
        
        # 添加过滤配置
        if filter_config:
            prompt += f"\n一些提示：\n{filter_config}\n"
        
        prompt += "\n请结合提供的文档以及用户对话历史，判断提供的文档是不是能和用户的最后一个问题相关。"
        prompt += "如果该文档提供的知识能够和用户的问题相关，那么请回复\"yes/<relevant>\" 否则回复\"no/<relevant>\"。"
        prompt += "其中， <relevant> 是你认为文档中和问题的相关度，0-10之间的数字，数字越大表示相关度越高。"

        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "你是一个文档相关性分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10  # 限制输出长度
            )
            #print("prompt:",prompt)
            #print("relevant score:",response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return None

class DocFilter:
    def __init__(
        self,
        api_key: str,
        args: AutoCoderArgs,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
        model_name: Optional[str] = None,   
    ):
        self.llm = OpenAIWrapper(api_key=api_key, base_url=base_url, model_name=model_name)
        self.args = args
        self.relevant_score = self.args.rag_doc_filter_relevance or 5
        self.path = path

    def filter_docs(
        self, conversations: List[Dict[str, str]], documents: List[SourceCode]
    ) -> List[FilterDoc]:
        return self.filter_docs_with_threads(conversations, documents)

    def filter_docs_with_threads(
        self, conversations: List[Dict[str, str]], documents: List[SourceCode]
    ) -> List[FilterDoc]:
        console = Console()
        if self.path:
            rag_manager = RagConfigManager(path=self.path)
            rag_config = rag_manager.load_config()
        else:
            rag_config = RagConfig()

        documents = list(documents)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pass
            #task = progress.add_task("[cyan]Filtering documents...", total=len(documents))

            with ThreadPoolExecutor(
                max_workers=self.args.index_filter_workers or 5
            ) as executor:
                future_to_doc = {}
                for doc in documents:
                    submit_time = time.time()

                    def _run(conversations, docs):
                        submit_time_1 = time.time()
                        try:
                            v = self.llm.check_relevance(
                                conversations=conversations,
                                documents=docs,
                                filter_config=rag_config.filter_config,
                            )
                        except Exception as e:
                            logger.error(f"Error in check_relevance: {str(e)}")
                            return (None, submit_time_1, time.time())

                        end_time_2 = time.time()
                        return (v, submit_time_1, end_time_2)

                    m = executor.submit(
                        _run,
                        conversations,
                        [f"##File: {doc.module_name}\n{doc.source_code}"],
                    )
                    future_to_doc[m] = (doc, submit_time)

                relevant_docs = []
                for future in as_completed(list(future_to_doc.keys())):
                    try:
                        doc, submit_time = future_to_doc[future]
                        end_time = time.time()
                        v, submit_time_1, end_time_2 = future.result()
                        task_timing = TaskTiming(
                            submit_time=submit_time,
                            end_time=end_time,
                            duration=end_time - submit_time,
                            real_start_time=submit_time_1,
                            real_end_time=end_time_2,
                            real_duration=end_time_2 - submit_time_1,
                        )
                        #progress.update(task, advance=1)

                        relevance = parse_relevance(v)
                        if (
                            relevance
                            and relevance.is_relevant
                            and relevance.relevant_score >= self.relevant_score
                        ):
                            relevant_docs.append(
                                FilterDoc(
                                    source_code=doc,
                                    relevance=relevance,
                                    task_timing=task_timing,
                                )
                            )
                    except Exception as exc:
                        logger.error(f"Document processing generated an exception: {exc}")

        # Sort relevant_docs by relevance score in descending order
        relevant_docs.sort(key=lambda x: x.relevance.relevant_score, reverse=True)
        return relevant_docs 