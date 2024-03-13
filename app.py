from colbert_vectorstore import Astra_ColBERT_VectorStore
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os 
import streamlit as st

@st.cache_resource
def init():
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
    return colbert_vstore, vstore


query = st.text_input('Query')
                      
colbert_vstore, vstore = init()

results = colbert_vstore.similarity_search(query)
dpr_results = vstore.similarity_search(query)

col1, col2 = st.columns(2)

with col1:
    st.header('ColBERT Results')
    st.json(results)

with col2:
    st.header('DPR Results')
    st.json(dpr_results)
