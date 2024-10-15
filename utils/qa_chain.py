# utils/qa_chain.py
import yaml
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from easydict import EasyDict 

def build_qa_chain(document_text, config_path='config/config.yaml'):

    with open(config_path, 'r', encoding='utf-8') as f:
        config = EasyDict(yaml.safe_load(f))
    embeddings_config = config.embeddings
    model_config = config.model
    vector_store_config = config.vector_store
    
    # 创建嵌入向量
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_config.model_name)
    vector_store = FAISS.from_texts(document_text, embeddings)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_config.name_or_path)
    
    # 创建 HuggingFace 的文本生成管道
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=2048
    )
    
    # 将 HuggingFace 管道包装为 Langchain 的 LLM
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # 使用 from_llm 方法构建问答链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    return qa_chain
