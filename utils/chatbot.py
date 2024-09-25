from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore(chunks):
    api_key = os.getenv("OPENAI_API_KEYS")

    if not chunks:
        raise ValueError("Os chunks de texto estão vazios. Verifique o processamento dos arquivos.")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def ensure_token_limit(chunks, max_tokens=16000):
    valid_chunks = []
    total_tokens = 0
    for chunk in chunks:
        chunk_tokens = len(chunk)
        if total_tokens + chunk_tokens < max_tokens:
            valid_chunks.append(chunk)
            total_tokens += chunk_tokens
        else:
            break
    return valid_chunks
