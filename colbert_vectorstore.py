from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from astrapy.db import AstraDB
from torch import tensor
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.indexing.collection_encoder import CollectionEncoder
import itertools

from typing import (
    Any,
    Iterable,
    List,
    cast,    
    Optional,    
)

class Astra_ColBERT_VectorStore(VectorStore):
    def __init__(
        self,
        *,        
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,        
        namespace: Optional[str] = None,        
    ) -> None:        
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.namespace = namespace
        if token and api_endpoint:
            self.astra_db = AstraDB(
                token=cast(str, self.token),
                api_endpoint=cast(str, self.api_endpoint),
                namespace=self.namespace,
            )
        if self.astra_db is not None:
            self.collection = self.astra_db.create_collection(self.collection_name)            
            self.collection_bert = self.astra_db.create_collection(self.collection_name + "_bert", dimension=128,metric='dot_product')
                
        # load colbert model
        cf = ColBERTConfig(checkpoint='./colbertv2.0')
        self.checkpoint = Checkpoint(cf.checkpoint, colbert_config=cf)
        self.encoder = CollectionEncoder(cf, self.checkpoint)        
    
    def maxsim(self,qv, document_embeddings):
        return max(qv @ dv for dv in document_embeddings)

    def score(self,query_embeddings, document_embeddings):
        return sum(self.maxsim(qv, document_embeddings) for qv in query_embeddings)

    def add_texts(self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any) -> List[str]:
        text_batch = 20
        for j in range (0, len(texts), text_batch):            
            embeddings_flat, counts = self.encoder.encode_passages(texts[j: j+text_batch])
            start_indices = [0] + list(itertools.accumulate(counts[:-1]))
            embeddings_by_part = [embeddings_flat[start:start+count] for start, count in zip(start_indices, counts)]
            batch = 20 
            for part, embeddings in enumerate(embeddings_by_part):        
                for i in range (0, len(embeddings), batch):
                    self.collection_bert.insert_many([{"part":part,"token": i,"$vector": e, 'metadata': metadatas[part]} for i, e in enumerate(embeddings[i:i+batch])])
            
            self.collection.insert_many( [ {"content": content, "part": part, "metadata":metadatas[part]} for part,content in enumerate(texts[j:j+text_batch]) ])
        return len(texts)
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        content = []
        metadata = []
        for doc in documents:
            content.append(doc.page_content)
            metadata.append(doc.metadata)
        return self.add_texts(content, metadata)
                
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        encode = lambda q: self.checkpoint.queryFromText([q])[0]
        query_encodings = encode(query)
        docparts = set()
        for qv in query_encodings:
            results = self.collection_bert.vector_find (qv, limit=k)            
            for r in results:               
                docparts.add((r['part'], r['metadata']['source']))        
        scores = {}
        for (part, source) in docparts:
            rows = self.collection_bert.find({"part":part, "metadata.source": source})            
            embeddings_for_part = [tensor(row['$vector']) for row in rows['data']['documents']]
            scores[(part,source)] = self.score(query_encodings, embeddings_for_part)        
        scores = sorted(scores, key=scores.get, reverse=True)[:5]
        results = []
        for (part, source) in scores:
            rows = self.collection.find({"part":part, "metadata.source": source})
            for row in rows['data']['documents']:
                results.append(Document(page_content=row[], metadata=row['metadata']))
        return results
    
    def get_embedding(self, doc_id: str) -> Embeddings:
        return super().get_embedding(doc_id)
       
    def from_texts(self, texts: Iterable[str]) -> List[Document]:
        return super().from_texts(texts)
    
    def from_documents(self, documents: List[Document]) -> List[str]:
        return super().from_documents(documents)
    
    def get_document(self, doc_id: str) -> Document:
        return super().get_document(doc_id)
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        return super().get_documents(doc_ids)