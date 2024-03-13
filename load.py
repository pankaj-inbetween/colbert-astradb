from colbert_vectorstore import Astra_ColBERT_VectorStore
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

colbert_vstore = Astra_ColBERT_VectorStore(    
        collection_name="interactions",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )
vstore = AstraDBVectorStore(
        embedding=OpenAIEmbeddings(),
        collection_name="papers",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )

loader = PyPDFDirectoryLoader("files/")
docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=3000,
    chunk_overlap=50))
colbert_vstore.add_documents(docs)
vstore.add_documents(docs)
