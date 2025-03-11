import faiss
from uuid import uuid4

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from agent.llm import embeddings


class Retriever:
    def __init__(self):
        self.document = None
        self.chunks = None
        self.embeddings = embeddings
        self.vector_store = None

    def load_document(self, path_to_document):
        loader = TextLoader(path_to_document)
        self.document = loader.load()

    def split_document (self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = text_splitter.split_documents(self.document)

    def create_vector_store(self):
        index = faiss.IndexFlatL2(len(embeddings.embed_query("Test")))
        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        uuids = [str(uuid4()) for _ in range(len(self.chunks))]
        self.vector_store.add_documents(documents=self.chunks, ids=uuids)
        return self.vector_store

    def save_db(self):
        self.vector_store.save_local("index")

    def load_db(self):
        self.vector_store = FAISS.load_local(
            "index", embeddings, allow_dangerous_deserialization=True
        )