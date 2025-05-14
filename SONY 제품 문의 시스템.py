import os
import torch
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline


# ✅ GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Tavily Web 검색 설정
os.environ["TAVILY_API_KEY"] = "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"
web_search_tool = TavilySearchResults(k=2)

# ✅ 벡터 저장소 설정
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-MiniLM-L3-v2',
    model_kwargs={'device': 'cpu'}  # GPU가 부족하면 cpu 사용
)
directory_path = 'SONY_Chatbot/SONY vectorstore 생성'
vectorstore = Chroma(persist_directory=directory_path, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# ✅ 유사도 모델 설정 (product name 필터용)
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ✅ LLM 설정 (OpenChat 3.5)
model_name = "openchat/openchat_3.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=gen_pipeline)

# ✅ LLMChain 프롬프트
prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a helpful assistant that summarizes and explains information in a simple and clear way.\n"
        "Based on the document or web search, please answer the following question clearly and concisely:\n\n"
        "{question}"
    )
)

chain = LLMChain(prompt=prompt, llm=llm)

# ✅ 응답 생성 함수
def get_answer_with_web_search(query, product_name):
    all_docs = retriever.get_relevant_documents(query)

    doc_titles = [
        doc.metadata.get("file_name", doc.page_content[:30]).split('.')[0]
        for doc in all_docs
    ] if all_docs else []

    product_embedding = similarity_model.encode(product_name)
    doc_embeddings = similarity_model.encode(doc_titles, batch_size=4) if doc_titles else []

    filtered_docs = [
        doc for doc, sim in zip(all_docs, cosine_similarity([product_embedding], doc_embeddings)[0])
        if sim >= 0.7
    ] if doc_embeddings else []

    # 문서 응답
    if filtered_docs:
        context = "\n".join([doc.page_content for doc in filtered_docs])
        full_prompt = f""" 당신은 기술 문서를 쉽게 요약하고 설명하는 도우미입니다. 아래 문서는 소니 제품 '{product_name}'에 대한 설명입니다.
        사용자 질문: "{query}"
        이 질문에 대해 문서를 바탕으로 핵심만 뽑아 이해하기 쉽게 정리해 주세요:
        문서 내용: {context} """
        doc_answer = chain.run({"question": full_prompt}).strip()

        source_info = "\n".join([
            f"출처 - 파일명: {doc.metadata.get('file_name', '알 수 없음')}, 페이지 번호: {doc.metadata.get('page_number', '미표기')}"
            for doc in filtered_docs
        ])
    else:
        doc_answer = "문서에서 관련 정보를 찾지 못했습니다."
        source_info = "출처 없음"

    # 웹 검색 수행
    web_results = web_search_tool.run(query)
    st.write("🔍 웹 검색 결과 (raw):", web_results)

    if not web_results:
        web_summary = "웹 검색 결과가 없습니다."
    else:
        web_context = ""
        for i, result in enumerate(web_results[:3]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            if snippet:
                web_context += f"{i+1}. {title} - {snippet}\n"

        web_prompt = f"다음은 '{query}'에 대한 웹 검색 결과입니다. 이를 바탕으로 간결한 요약을 제공해주세요:\n\n{web_context}"
        web_summary = chain.run({"question": web_prompt}).strip()

    combined_answer = doc_answer + "\n\n📡 웹 검색 요약:\n" + web_summary
    return combined_answer, source_info

# ✅ Streamlit UI
st.title("소니 제품 설명서 챗봇")
product_name = st.selectbox(
    "제품명을 선택하세요:",
    [
        "글래스 사운드 스피커 LSPX-S3",
        "디지털 카메라 A7",
        "렌즈 교환 가능 디지털 카메라 ILCE-7CM2",
        "무선 노이즈 제거 스테레오 헤드셋 WF-1000XM5",
        "무선 노이즈 제거 스테레오 헤드셋 WH-1000XM5",
        "무선 스테레오 헤드셋 Float Run",
        "무선 스피커 SRS-XE300"
    ]
)
query = st.text_input("제품 관련 질문을 입력하세요:")
if st.button("질문하기"):
    with st.spinner("답변을 생성 중입니다..."):
        answer, source_info = get_answer_with_web_search(query, product_name)
    st.subheader("📩 질문")
    st.write(query)
    st.subheader("💬 답변")
    st.markdown(answer)
    st.subheader("📎 출처")
    st.markdown(source_info)



# def get_answer_with_web_search(query, product_name):
#     all_docs = retriever.get_relevant_documents(query)
#     product_embedding = similarity_model.encode(product_name)
#     doc_embeddings = similarity_model.encode(
#         [doc.metadata.get("file_name", "").split('.')[0] for doc in all_docs]
#     )

#     filtered_docs = [
#         doc for doc, sim in zip(all_docs, cosine_similarity([product_embedding], doc_embeddings)[0])
#         if sim >= 0.7
#     ]

#     # 문서 없으면 → 웹 검색
#     if not filtered_docs:
#         web_results = web_search_tool.invoke(query)
#         snippets = [result.get('snippet', '결과 없음') for result in web_results]
#         if snippets:
#             return (
#                 "문서에서 찾지 못한 결과를 웹 검색에서 반환합니다:\n" + "\n".join(snippets),
#                 "출처: 웹 검색"
#             )
#         else:
#             return "문서 및 웹 검색에서 관련 정보를 찾을 수 없습니다.", "출처 없음"

#     # 문서 기반 응답 생성
#     context = "\n".join([doc.page_content for doc in filtered_docs])
#     prompt = f"문서 내용을 바탕으로 질문 '{query}'에 답해주세요: {context}"
#     answer = llm(prompt)

#     source_info = "\n".join([
#         f"출처 - 파일명: {doc.metadata.get('file_name')}, 페이지 번호: {doc.metadata.get('page_number')}"
#         for doc in filtered_docs
#     ])

#     return answer, source_info

# # AI 응답 생성 함수
# def get_answer_with_web_search(query, product_name):
#     all_docs = retriever.get_relevant_documents(query)
#     product_embedding = similarity_model.encode(product_name)
#     doc_embeddings = similarity_model.encode(
#         [doc.metadata.get("file_name", "").split('.')[0] for doc in all_docs]
#     )
#     filtered_docs = [
#         doc for doc, sim in zip(all_docs, cosine_similarity([product_embedding], doc_embeddings)[0])
#         if sim >= 0.7
#     ]

#     if not filtered_docs:
#         web_results = web_search_tool.invoke(query)
#         snippets = [result.get('snippet', '결과 없음') for result in web_results]
#         if snippets:
#             return (
#                 "문서에서 찾지 못한 결과를 웹 검색에서 반환합니다:\n" + "\n".join(snippets),
#                 "출처: 웹 검색"
#             )
#         else:
#             return "문서 및 웹 검색에서 관련 정보를 찾을 수 없습니다.", "출처 없음"

#     context = "\n".join([doc.page_content for doc in filtered_docs])
#     inputs = tokenizer(
#         f"문서 내용을 바탕으로 질문 '{query}'에 답해주세요: {context}",
#         return_tensors="pt",
#         truncation=True,
#         max_length=512
#     ).to(device)

#     outputs = model.generate(inputs["input_ids"], max_new_tokens=200, do_sample=True)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#     source_info = "\n".join([
#         f"출처 - 파일명: {doc.metadata.get('file_name')}, 페이지 번호: {doc.metadata.get('page_number')}"
#         for doc in filtered_docs
#     ])

#     return answer, source_info

# # Streamlit 앱
# if "page" not in st.session_state:
#     st.session_state.page = "loading"
#     set_background(loading_image_path)
#     st.write("로딩 중입니다... 잠시만 기다려 주세요!")
#     st.button("시작", on_click=lambda: st.session_state.update({"page": "input"}))
# else:
#     if st.session_state.page == "input":
#         set_background(input_image_path)
#         st.title("소니 제품 문의 시스템")
#         product_name = st.selectbox(
#             "제품명을 선택하세요:",
#             ["글래스 사운드 스피커 LSPX-S3", "디지털 카메라 A7", "렌즈 교환 가능 디지털 카메라 ILCE-7CM2", "무선 노이즈 제거 스테레오 헤드셋 WF-1000XM5", "무선 노이즈 제거 스테레오 헤드셋 WH-1000XM5", "무선 스테레오 헤드셋 Float Run", "무선 스피커 SRS-XE300" ]
#         )
#         query = st.text_input("문의 사항을 입력하세요:")
#         if st.button("질문하기"):
#             with st.spinner("응답을 생성 중입니다..."):
#                 answer, source_info = get_answer_with_web_search(query, product_name)
#             st.subheader("질문")
#             st.write(query)
#             st.subheader("답변")
#             st.write(answer)
#             st.subheader("출처 정보")
#             st.write(source_info)
