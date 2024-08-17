from pydantic import BaseModel, Field
from typing import Any, List, Tuple, Optional
import pandas as pd
from langchain_core.retrievers import BaseRetriever

class HybridRFFRetriever(BaseRetriever):
    semantic_retriever: Any = Field(...)
    lexical_retriever: Any = Field(...)
    top_k: int = Field(default=4)
    rrf_weight: int = Field(default=60)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[Any] = None) -> List[Any]:
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)
        lexical_docs = self.lexical_retriever.get_relevant_documents(query)
        max_docs = max(len(semantic_docs), len(lexical_docs))
        semantic_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(semantic_docs)] + [''] * (max_docs - len(semantic_docs))
        lexical_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(lexical_docs)] + [''] * (max_docs - len(lexical_docs))
        semantic_scores = [doc.metadata.get('score', 0) for doc in semantic_docs] + [0] * (max_docs - len(semantic_docs))
        lexical_scores = [doc.metadata.get('score', 0) for doc in lexical_docs] + [0] * (max_docs - len(lexical_docs))
        df = pd.DataFrame({
            'semantic_id': semantic_ids,
            'lexical_id': lexical_ids,
            'semantic_score': semantic_scores,
            'lexical_score': lexical_scores
        })
        df['rrf_id'], df['rrf_score'] = zip(*df.apply(
            lambda row: self.rrf_pure(
                (row['semantic_id'], row['lexical_id']),
                (row['semantic_score'], row['lexical_score']),
                self.rrf_weight,
                self.top_k
            ), axis=1
        ))
        df = df.sort_values(by='rrf_score', ascending=False)
        top_ids = df['rrf_id'].tolist()[:self.top_k]
        id_to_doc = {doc.metadata.get('doc_id', i): doc for i, doc in enumerate(semantic_docs + lexical_docs)}
        return [id_to_doc[id] for id in top_ids if id in id_to_doc and id != '']

    def rrf_pure(self, ids: Tuple[str, str], scores: Tuple[float, float], rrf_k: int, top_k: int) -> Tuple[str, float]:
        df = pd.Series({ids[i]: scores[i] for i in range(len(ids))})
        rank_df = df.rank(ascending=False, method='min').fillna(0)
        rank_df = rank_df.apply(lambda r: 1 / (r + rrf_k) if r > 0 else 0)
        rank_df = rank_df.sort_values(ascending=False)
        return rank_df.index[0], rank_df.iloc[0]
