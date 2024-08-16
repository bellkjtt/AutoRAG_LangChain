import pdfplumber
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import unicodedata
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF

# Langchain 관련
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(file_path, chunk_size=1500, chunk_overlap=200):
    """PDF 텍스트 추출 후 chunk 단위로 나누기"""
    # PDF 파일 열기
    doc = fitz.open(file_path)
    text = ''
    # 모든 페이지의 텍스트 추출
    for page in doc:
        text += page.get_text()
    # 텍스트를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_temp = splitter.split_text(text)
    # Document 객체 리스트 생성
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks

def extract_tables_from_pdf(pdf_path: str) -> List[pd.DataFrame]:
    """PDF에서 표를 추출하여 pandas DataFrame 리스트로 반환합니다."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

def dataframe_to_html(df: pd.DataFrame) -> str:
    """pandas DataFrame을 HTML 표로 변환합니다."""
    html = df.to_html(index=False)
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    try:
        df = pd.read_html(str(table))[0]
        return df.to_markdown(index=False)
    except:
        return str(table)

def process_pdf_to_faiss(pdf_path: str, model_path: str = "intfloat/multilingual-e5-base") -> FAISS:
    """PDF에서 표를 추출하고, HTML로 변환한 후 FAISS DB에 저장합니다."""
    # 1. PDF에서 표 추출
    tables = extract_tables_from_pdf(pdf_path)
    
    # 2. 표를 HTML로 변환
    html_tables = [dataframe_to_html(df) for df in tables]
    
    # 3. HTML 텍스트를 Document 형태로 변환 (FAISS 인덱싱을 위해)
    chunks = [Document(page_content=table, metadata={"priority": 1, "page_type":"table"}) for table in html_tables]
    
    return chunks

# 사용 예시

def create_vector_db(chunks, model_path="intfloat/multilingual-e5-base"):
    """FAISS DB 생성"""
    # 임베딩 모델 설정
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # FAISS DB 생성 및 반환
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

def normalize_path(path):
    """경로 유니코드 정규화"""
    return unicodedata.normalize('NFC', path)

def process_pdfs_from_dataframe(df, base_directory):
    """딕셔너리에 pdf명을 키로해서 DB, retriever 저장"""
    pdf_databases = {}
    unique_paths = df['Source_path'].unique()
    
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")
        
        # PDF 처리 및 벡터 DB 생성
        chunks = process_pdf(full_path)
        
        # BM25 Retriever 초기화
        okt_bm25 = BM25Retriever.from_documents(chunks, preprocess_func=okt_tokenize, search_kwargs={'k': 1})
        
        # FAISS DB 생성 및 Retriever 초기화
        db = create_vector_db(chunks)
        faiss = db.as_retriever(search_kwargs={'k': 3})
        
        # HybridCCRetriever 생성
        hybrid_cc_retriever = HybridCCRetriever(
            semantic_retriever=faiss,
            lexical_retriever=okt_bm25,
            weight=0.81,  # 의미적 검색에 더 높은 가중치 부여
            normalize_method='tmm',
            top_k=3
        )
        
        # 결과 저장
        pdf_databases[pdf_title] = {
            'db': db,
            'retriever': hybrid_cc_retriever
        }
    return pdf_databases

