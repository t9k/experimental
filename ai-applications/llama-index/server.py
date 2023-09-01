import asyncio
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.llms import HuggingFaceLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import StorageContext
from llama_index.text_splitter import TokenTextSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

QDRANT_SERVER_URI = 'http://qdrant-db:6333'

MODEL_PATH = './Baichuan-13B-Chat'
EMBEDDING_PATH = './all-mpnet-base-v2'
INPUT_FILE = 'docs/比亚迪：2023年半年度报告.pdf'


async def homepage(request):
    payload = await request.body()
    string = payload.decode('utf-8')
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    model_path = os.getenv('MODEL_PATH', MODEL_PATH)
    embedding_path = os.getenv('EMBEDDING_PATH', EMBEDDING_PATH)

    # Prepare storage
    qdrant_client = QdrantClient(QDRANT_SERVER_URI)
    vector_store = QdrantVectorStore(client=qdrant_client,
                                     collection_name='demo')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Prepare models
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_fast=False,
                                              trust_remote_code=True)
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={
            'num_beams': 1,
            'do_sample': True,
            'top_p': 0.8,
            'temperature': 0.8
        },
        tokenizer=tokenizer,
        model=model,
        device_map='auto',
        model_kwargs={
            'torch_dtype': torch.float16,
            'load_in_8bit': True,
        },
    )
    embed_model = HuggingFaceEmbeddings(model_name=embedding_path)
    service_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                   llm=llm)

    # Define index
    documents = SimpleDirectoryReader(input_files=[INPUT_FILE]).load_data()
    text_splitter = TokenTextSplitter(separator='\n',
                                      chunk_size=512,
                                      chunk_overlap=64)
    parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes=nodes,
                             storage_context=storage_context,
                             service_context=service_context)

    # Define query engine
    text_qa_template = PromptTemplate('''
    上下文信息如下：
    ---------------------
    {context_str}
    ---------------------
    根据上下文信息而非已有的知识，回答查询。回答需要遵守以下规则：
    - 生成易于理解的输出，避免生成无意义的文本输出。
    - 若源文档提供的信息不足以回答问题，则如实阐述，不要凭空捏造。
    - 仅生成所请求的输出，不要在所请求的输出之前或之后包含任何其他语言。
    - 不要说“谢谢”、“乐于帮助”、“是一个 AI 代理”等，也不要说“根据提供的信息”，“根据上下文信息”等。请直接回答问题。
    - 生成规范、专业的中文。
    - 不生成冒犯性或粗俗的语言。

    查询：{query_str}
    回答：
    ''')

    refine_template = PromptTemplate('''
    原始查询如下：{query_str}
    我们已提供现有答案：{existing_answer}
    我们有机会通过下面给出的更多上下文来完善现有答案（仅在需要时）。
    ----------------
    {context_msg}
    ----------------
    基于新上下文，完善原始答案以更好地回答查询。
    如果上下文无用，请返回原始答案。
    完善后的答案：
    ''')

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        text_qa_template=text_qa_template,
        refine_template=refine_template)
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.4)]
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors)

    # Q&A
    while True:
        (question, response_q) = await q.get()
        response = query_engine.query(question)
        await response_q.put(str(response))


app = Starlette(routes=[
    Route('/', homepage, methods=['POST']),
], )


@app.on_event('startup')
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))
