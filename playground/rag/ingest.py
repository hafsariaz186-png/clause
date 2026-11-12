from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


DATA_PATH = "shop_handbook.pdf"
CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:1b"
EMBED_PROVIDER = "ollama"
EMBED_MODEL = "nomic-embed-text:latest"
EMBED_DIM = 768

COLLECTION_NAME = "shop_handbook"
QDRANT_DATA = "./vector_store"


SYSTEM = """
You are a helpful AI clause shop assistant!
"""

client = QdrantClient(path=QDRANT_DATA)

# Indexing
# 1.Load
pages = PyPDFLoader(file_path=DATA_PATH).load_and_split()

# 2. Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents=pages)

# 3. Embed
embed = init_embeddings(
    provider=EMBED_PROVIDER,
    model=EMBED_MODEL,
)

# 4. store

if client.collections_exits(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embed
)

vector_store.add_documents(documents=chunks)
