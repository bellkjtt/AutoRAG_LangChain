from pydantic import BaseModel, Field
from typing import Any, List, Tuple
import pandas as pd

class HybridRFFRetriever(BaseModel):
    semantic_retriever: Any = Field(...)
    lexical_retriever: Any = Field(...)
    top_k: int = Field(default=4)
    rrf_weight: int = Field(default=60)

    def _get_relevant_documents(self, query: str) -> List[Any]:
        # Semantic과 lexical retriever에서 관련 문서를 가져옴
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)
        lexical_docs = self.lexical_retriever.get_relevant_documents(query)

        # 문서 ID와 점수 가져오기
        semantic_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(semantic_docs)]
        lexical_ids = [doc.metadata.get('doc_id', i) for i, doc in enumerate(lexical_docs)]
        
        # 문서 ID와 점수를 데이터프레임으로 변환
        id_df = pd.DataFrame({
            'semantic_id': semantic_ids,
            'lexical_id': lexical_ids
        })
        score_df = pd.DataFrame({
            'semantic_score': [doc.metadata.get('score', 0) for doc in semantic_docs],
            'lexical_score': [doc.metadata.get('score', 0) for doc in lexical_docs]
        })

        # ID와 점수 데이터프레임 병합
        df = pd.concat([id_df, score_df], axis=1)

        # RRF 적용
        df['rrf_id'], df['rrf_score'] = zip(*df.apply(
            lambda row: self.rrf_pure(
                (row['semantic_id'], row['lexical_id']),
                (row['semantic_score'], row['lexical_score']),
                self.rrf_weight,
                self.top_k
            ), axis=1
        ))

        # 상위 K개 문서 선택 및 중복 제거
        df = df.sort_values(by='rrf_score', ascending=False)
        top_ids = df['rrf_id'].tolist()[:self.top_k]

        # 문서 결합 및 반환
        id_to_doc = {doc.metadata.get('doc_id', i): doc for i, doc in enumerate(semantic_docs + lexical_docs)}
        return [id_to_doc[id] for id in top_ids if id in id_to_doc]

    def rrf_pure(self, ids: Tuple[str, str], scores: Tuple[float, float], rrf_k: int, top_k: int) -> Tuple[str, float]:
        df = pd.Series({ids[i]: scores[i] for i in range(len(ids))})
        rank_df = df.rank(ascending=False, method='min').fillna(0)
        rank_df['rrf'] = rank_df.apply(lambda r: 1 / (r + rrf_k) if r > 0 else 0)
        rank_df = rank_df.sort_values(by='rrf', ascending=False)
        return rank_df.index[0], rank_df['rrf'].iloc[0]

    # Runnable의 invoke() 메서드를 구현하여 호환 가능하게 만들기
    def invoke(self, query: str) -> List[Any]:
        return self._get_relevant_documents(query)

    # __call__도 구현해서 직접 호출 가능하게 설정
    def __call__(self, query: str) -> List[Any]:
        return self.invoke(query)
