from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from core.database import get_all_papers
from core.vector_store import get_vector_store
from core.rag_pipeline import RAGPipeline

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    paper_ids: Optional[List[int]] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str
    paper_ids: Optional[List[int]] = None

@router.get("/papers")
def list_papers():
    return get_all_papers()

@router.post("/search")
def search_papers(req: SearchRequest):
    try:
        vs = get_vector_store()
        results = vs.hybrid_search(req.query, k=req.top_k, paper_ids=req.paper_ids)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        pipeline = RAGPipeline()
        result = pipeline.answer(
            query=req.query,
            paper_ids=req.paper_ids,
            chat_history=[]
        )
        return {
            "session_id": req.session_id,
            "response": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
