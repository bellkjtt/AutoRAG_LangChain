from pydantic import BaseModel, Field
from typing import Any, List, Optional
import pandas as pd
from langchain_core.retrievers import BaseRetriever

class HybridCCRetriever(BaseRetriever):
    semantic_retriever: Any = Field(...)
    lexical_retriever: Any = Field(...)
    weight: float = Field(default=0.5)
    normalize_method: str = Field(default='tmm')
    top_k: int = Field(default=4)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Any]:
        # Semantic과 lexical retriever에서 관련 문서를 가져옵니다.
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)
        lexical_docs = self.lexical_retriever.get_relevant_documents(query)
        
        # 문서 ID와 점수 가져오기
        semantic_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(semantic_docs)]
        lexical_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(lexical_docs)]
        
        # Semantic과 Lexical 점수 정규화 및 병합
        df = pd.DataFrame({
            'semantic': pd.Series(dict(zip(semantic_ids, [doc.metadata.get('score', 0) for doc in semantic_docs]))),
            'lexical': pd.Series(dict(zip(lexical_ids, [doc.metadata.get('score', 0) for doc in lexical_docs]))),
        }).fillna(0)
        
        # 가중합 계산 및 상위 K개 문서 선택
        df['weighted_sum'] = df.mul((self.weight, 1.0 - self.weight)).sum(axis=1)
        df = df.sort_values(by='weighted_sum', ascending=False)
        top_ids = df.index.tolist()[:self.top_k]
        
        # 문서 결합 및 중복 제거
        id_to_doc = {doc.metadata.get('doc_id', i): doc for i, doc in enumerate(semantic_docs + lexical_docs)}
        return [id_to_doc[id] for id in top_ids if id in id_to_doc]
