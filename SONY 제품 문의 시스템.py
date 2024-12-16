import os
import streamlit as st
import base64
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.tools.tavily_search import TavilySearchResults

# 이미지 파일 경로
loading_image_path = '/home/work/DY/SONY/001.jpg'
input_image_path = '/home/work/DY/SONY/002.jpg'

# CSS 설정 함수
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# KoAlpaca 모델 설정 (로컬 경로 사용)
model_name = "/home/work/DY/KoAlpaca-llama-1-7b"

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 자동으로 GPU/CPU로 분배
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 적절한 데이터 타입 설정
        offload_folder="/tmp/model_offload"  # 오프로드 위치 설정
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# 웹 검색 설정
os.environ["TAVILY_API_KEY"] = "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"
web_search_tool = TavilySearchResults(k=2)

# 벡터 저장소 설정
directory_path = '/home/work/DY/SONY/vectorstore'
embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': device})
vectorstore = Chroma(persist_directory=directory_path, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

# 유사도 모델 설정
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# AI 응답 생성 함수
def get_answer_with_web_search(query, product_name):
    all_docs = retriever.get_relevant_documents(query)
    product_embedding = similarity_model.encode(product_name)
    doc_embeddings = similarity_model.encode(
        [doc.metadata.get("file_name", "").split('.')[0] for doc in all_docs]
    )
    filtered_docs = [
        doc for doc, sim in zip(all_docs, cosine_similarity([product_embedding], doc_embeddings)[0])
        if sim >= 0.7
    ]

    if not filtered_docs:
        web_results = web_search_tool.invoke(query)
        snippets = [result.get('snippet', '결과 없음') for result in web_results]
        if snippets:
            return (
                "문서에서 찾지 못한 결과를 웹 검색에서 반환합니다:\n" + "\n".join(snippets),
                "출처: 웹 검색"
            )
        else:
            return "문서 및 웹 검색에서 관련 정보를 찾을 수 없습니다.", "출처 없음"

    context = "\n".join([doc.page_content for doc in filtered_docs])
    inputs = tokenizer(
        f"문서 내용을 바탕으로 질문 '{query}'에 답해주세요: {context}",
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(inputs["input_ids"], max_new_tokens=200, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    source_info = "\n".join([
        f"출처 - 파일명: {doc.metadata.get('file_name')}, 페이지 번호: {doc.metadata.get('page_number')}"
        for doc in filtered_docs
    ])

    return answer, source_info

# Streamlit 앱
if "page" not in st.session_state:
    st.session_state.page = "loading"
    set_background(loading_image_path)
    st.write("로딩 중입니다... 잠시만 기다려 주세요!")
    st.button("시작", on_click=lambda: st.session_state.update({"page": "input"}))
else:
    if st.session_state.page == "input":
        set_background(input_image_path)
        st.title("소니 제품 문의 시스템")
        product_name = st.selectbox(
            "제품명을 선택하세요:",
            ["글래스 사운드 스피커 LSPX-S3", "디지털 카메라 A7", "렌즈 교환 가능 디지털 카메라 ILCE-7CM2", "무선 노이즈 제거 스테레오 헤드셋 WF-1000XM5", "무선 노이즈 제거 스테레오 헤드셋 WH-1000XM5", "무선 스테레오 헤드셋 Float Run", "무선 스피커 SRS-XE300" ]
        )
        query = st.text_input("문의 사항을 입력하세요:")
        if st.button("질문하기"):
            with st.spinner("응답을 생성 중입니다..."):
                answer, source_info = get_answer_with_web_search(query, product_name)
            st.subheader("질문")
            st.write(query)
            st.subheader("답변")
            st.write(answer)
            st.subheader("출처 정보")
            st.write(source_info)
