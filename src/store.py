"""Local ChromaDB + local SentenceTransformer embeddings."""
import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMA_DIR, EMBEDDING_MODEL

class TranscriptStore:
    def __init__(self):
        self.model=SentenceTransformer(EMBEDDING_MODEL)
        self.client=chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection=self.client.get_or_create_collection(
            name="ami_meetings", metadata={"hnsw:space":"cosine"}
        )
    def reset(self):
        try: self.client.delete_collection("ami_meetings")
        except Exception: pass
        self.collection=self.client.get_or_create_collection(
            name="ami_meetings", metadata={"hnsw:space":"cosine"}
        )
    def add_records(self, records, batch_size=64):
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            texts=[x["text"] for x in batch]
            embeddings=self.model.encode(texts, normalize_embeddings=True).tolist()
            self.collection.upsert(
                ids=[x["chunk_id"] for x in batch], documents=texts,
                embeddings=embeddings,
                metadatas=[{k:v for k,v in x.items() if k!="text"} for x in batch]
            )
    def search(self, query, team=None, n_results=8, meeting_ids=None):
        where=None
        clauses=[]
        if team: clauses.append({"team":team})
        if meeting_ids: clauses.append({"meeting_id":{"$in":meeting_ids}})
        if len(clauses)==1: where=clauses[0]
        elif len(clauses)>1: where={"$and":clauses}
        emb=self.model.encode([query], normalize_embeddings=True).tolist()
        count=self.collection.count()
        if not count: return []
        res=self.collection.query(query_embeddings=emb,n_results=min(n_results,count),where=where,
                                  include=["documents","metadatas","distances"])
        out=[]
        for doc,meta,dist in zip(res["documents"][0],res["metadatas"][0],res["distances"][0]):
            out.append({"text":doc,**meta,"distance":float(dist)})
        return out
