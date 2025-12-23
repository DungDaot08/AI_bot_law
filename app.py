from fastapi.staticfiles import StaticFiles
import os
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# =====================
# CONFIG (PHẢI TRÙNG load_and_index.py)
# =====================
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "legal_docs"
EMBED_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY")
# =====================
# INIT
# =====================
app = FastAPI(title="Vietnam Legal QA Chatbot")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0,
)

# =====================
# SCHEMA
# =====================


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 6


class Source(BaseModel):
    law_name: str
    article: str
    clause: str
    file: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source]


# =====================
# 1. AI REWRITE (CHUẨN HÓA CÂU HỎI)
# =====================
def rewrite_question(question: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia pháp luật Việt Nam.\n"
                "Hãy chuẩn hóa câu hỏi sau thành câu hỏi pháp lý ngắn gọn,\n"
                "trung tính, phù hợp để tra cứu văn bản pháp luật.\n"
                "KHÔNG trả lời, KHÔNG giải thích."
            )
        ),
        HumanMessage(content=question),
    ]

    return llm.invoke(prompt).content.strip()


# =====================
# 2. SEARCH + FILTER
# =====================
def search_docs(query: str, k: int):
    docs = vectordb.similarity_search(query, k=k)

    # ❗ LOẠI BỎ CHUNK KHÔNG CÓ GIÁ TRỊ PHÁP LÝ
    filtered = []
    for d in docs:
        art = d.metadata.get("article", "").lower()
        if any(x in art for x in ["thi hành", "trách nhiệm", "hiệu lực", "phụ lục"]):
            continue
        filtered.append(d)

    return filtered


# =====================
# 3. BUILD ANSWER (CHỈ DÙNG CONTEXT)
# =====================
def build_answer(question: str, docs) -> Dict:
    if not docs:
        return {
            "answer": "Không tìm thấy quy định phù hợp trong các văn bản pháp luật đã được cung cấp.",
            "sources": [],
        }

    context = ""
    sources = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        context += f"\n[{i}] {doc.page_content}\n"

        sources.append(
            {
                "law_name": meta.get("law_name", ""),
                "article": meta.get("article", ""),
                "clause": meta.get("clause", ""),
                "file": meta.get("source", ""),
            }
        )

    prompt = [
        SystemMessage(
            content=(
                "Bạn là trợ lý pháp luật Việt Nam.\n"
                "CHỈ sử dụng nội dung pháp luật được cung cấp.\n"
                "KHÔNG suy đoán, KHÔNG dùng kiến thức bên ngoài.\n"
                "KHÔNG từ chối trả lời nếu có quy định trong dữ liệu.\n"
                "Nếu nội dung chưa đủ rõ, hãy trả lời theo đúng phạm vi điều luật."
            )
        ),
        HumanMessage(
            content=f"""
CÂU HỎI:
{question}

NỘI DUNG PHÁP LUẬT:
{context}

YÊU CẦU:
- Trả lời trực tiếp vào vấn đề
- Văn phong pháp luật, dễ hiểu
- Không liệt kê lan man
"""
        ),
    ]

    answer = llm.invoke(prompt).content.strip()

    return {
        "answer": answer,
        "sources": sources,
    }


# =====================
# API ENDPOINT
# =====================
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    # 1. Rewrite
    rewritten_question = rewrite_question(request.question)

    # 2. Search
    docs = search_docs(rewritten_question, request.top_k)

    # 3. Answer
    return build_answer(request.question, docs)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
