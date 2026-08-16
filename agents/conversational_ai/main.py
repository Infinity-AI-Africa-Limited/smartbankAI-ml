"""
Agent 6: Conversational AI Agent
Architecture: RAG (LangChain + ChromaDB) + Claude API for generation
Port: 8007
"""
import time
import logging
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Depends
import sys
sys.path.append("/app")

from shared.schemas.base import ConversationalRequest, ConversationalResponse, AgentResponse, HealthResponse
from shared.middleware.auth import verify_service_token, audit_log_middleware
from shared.utils.config import get_settings
from agents.conversational_ai.safety import safety_response

logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="SmartBank AI — Conversational AI Agent", version="1.0.0")
app.middleware("http")(audit_log_middleware)

retriever = None
llm_client = None
synthetic_retriever = None
_start_time = time.time()

SYSTEM_PROMPT = """You are SmartBank AI Assistant, a helpful and knowledgeable banking assistant 
for Nigerian bank customers. You help with account inquiries, transaction questions, product 
information, and financial guidance. You are powered by SmartBank AI, built by Infinity AI Africa Limited.

Rules:
- Always respond in the language the customer uses (English or Nigerian Pidgin)
- Never share another customer's information
- For transactions above ₦1,000,000, always recommend the customer visit a branch or call the helpline
- Cite CBN regulations when relevant
- If unsure, say so and offer to escalate to a human agent
- Keep responses concise and actionable

Context from knowledge base:
{context}
"""


@app.on_event("startup")
async def setup_rag():
    global retriever, llm_client, synthetic_retriever
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from anthropic import Anthropic

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(
            collection_name="smartbank_knowledge",
            embedding_function=embeddings,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("RAG retriever initialised")

        if settings.anthropic_api_key:
            llm_client = Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic client initialised")
    except Exception as e:
        logger.warning("RAG setup failed (running in stub mode): %s", e)
    baseline_path = Path(settings.model_dir) / "tfidf_retriever.pkl"
    if baseline_path.exists():
        with baseline_path.open("rb") as handle:
            synthetic_retriever = pickle.load(handle)
        logger.info("Synthetic retrieval baseline loaded")


def retrieve_context(query: str) -> tuple[str, list[str]]:
    if retriever is None:
        if synthetic_retriever is None:
            return "", []
        vectorizer = synthetic_retriever["vectorizer"]
        matrix = synthetic_retriever["matrix"]
        articles = synthetic_retriever["articles"]
        scores = (vectorizer.transform([query]) @ matrix.T).toarray().ravel()
        selected = np.argsort(scores)[::-1][:3]
        selected_articles = [articles[int(index)] for index in selected if scores[int(index)] > 0]
        return "\n\n".join(article["content"] for article in selected_articles), [article["source_id"] for article in selected_articles]
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [d.metadata.get("source", "SmartBank Knowledge Base") for d in docs]
    return context, sources


def generate_response(message: str, history: list[dict], context: str) -> str:
    if llm_client is None:
        if context:
            return f"Development knowledge-base response: {context} This information is advisory only; no banking action will be taken without your confirmation."
        return (
            f"Thank you for your message. I'm SmartBank AI Assistant. "
            f"You asked: '{message}'. Our team is here to help with all your banking needs. "
            f"For urgent matters, please call our 24/7 helpline: 0800-SMARTBANK."
        )

    messages = []
    for h in history[-6:]:  # last 3 turns
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    response = llm_client.messages.create(
        model=settings.llm_model,
        max_tokens=512,
        system=SYSTEM_PROMPT.format(context=context or "No specific context available."),
        messages=messages,
    )
    return response.content[0].text


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="conversational_ai", version="1.0.0",
        model_loaded=llm_client is not None or synthetic_retriever is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/chat", response_model=AgentResponse, dependencies=[Depends(verify_service_token)])
async def chat(req: ConversationalRequest):
    start = time.monotonic()
    guarded_response = safety_response(req.message)
    context, sources = retrieve_context(req.message) if guarded_response is None else ("", [])
    response_text = guarded_response or generate_response(req.message, req.conversation_history, context)

    suggested_actions = []
    msg_lower = req.message.lower()
    if any(w in msg_lower for w in ["transfer", "send money"]):
        suggested_actions.append("Go to Transfer")
    if any(w in msg_lower for w in ["loan", "borrow", "credit"]):
        suggested_actions.append("Apply for Loan")
    if any(w in msg_lower for w in ["balance", "account"]):
        suggested_actions.append("View Account")

    latency = (time.monotonic() - start) * 1000
    return AgentResponse(
        agent="conversational_ai", version="1.0.0", latency_ms=round(latency, 2),
        payload=ConversationalResponse(
            session_id=req.session_id,
            response=response_text,
            sources=sources,
            suggested_actions=suggested_actions,
            confidence=0.92 if llm_client else 0.5,
        ).model_dump() | {"human_review_required": True, "synthetic_knowledge_base": synthetic_retriever is not None},
    )
