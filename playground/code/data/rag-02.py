import os
from dotenv import load_dotenv
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

# from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
# Qrant


HANDBOOK_PATH = Path(__file__).resolve().parent / "data" / "shop_handbook.pdf"
load_dotenv()

# indexing
# 1. Load
pages = PyPDFLoader(str(HANDBOOK_PATH)).load() # list of Document objects

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents=pages)

# 3. Embed
embeddings = init_embeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    provider=os.getenv("EMBEDDING_PROVIDER")
)

# 4. Store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)


# RAG
# LLM
llm = init_chat_model(
    model=os.getenv(key="CHAT_MODEL"),
    model_provider=os.getenv(key="CHAT_PROVIDER"),
    api_key=os.getenv(key="CHAT_API_KEY") or None,
    temperature=0.1
)

query = input("You: ").strip()

# Retrieve
# retrieved_docs = vector_store.similarity_search(query. 3) # [Documents, ...]
# test =[]
# for doc in retrieved:
      texts.append(doc.page_content)
# context = "\n\n".join(texts) # string
