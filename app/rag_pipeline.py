import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from evaluation.evaluator import evaluate_answer
from models.providers import get_chat_model
from utils.config import AppConfig
from utils.history import save_qa_record
from utils.logger import get_logger
from utils.vector_store import retrieve_relevant_chunks


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are an intelligent document assistant.
Answer only from the provided document context.
If the context does not contain the answer, say: "I do not know based on the uploaded documents."
Be concise, cite relevant source names, and avoid unsupported claims.
"""


def _format_context(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "[Source {index}] File: {file_name}, Page: {page}, Relevance: {score:.3f}\n{content}".format(
                index=index,
                file_name=source.get("source", "Unknown"),
                page=source.get("page", "Unknown"),
                score=float(source.get("score", 0.0)),
                content=source.get("content", ""),
            )
        )
    return "\n\n".join(blocks)


def answer_question(question: str, config: AppConfig) -> dict[str, Any]:
    """Run retrieval, generation, source scoring, evaluation, and persistence."""
    total_start = time.perf_counter()
    retrieval = retrieve_relevant_chunks(question, config)
    sources = retrieval["sources"]

    if not sources:
        answer = "I do not know based on the uploaded documents."
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Question:\n{question}\n\nDocument context:\n{context}\n\nAnswer:",
                ),
            ]
        )
        chain = prompt | get_chat_model(config) | StrOutputParser()
        answer = chain.invoke(
            {
                "question": question,
                "context": _format_context(sources),
            }
        )

    response_time_seconds = time.perf_counter() - total_start
    metrics = evaluate_answer(
        question=question,
        answer=answer,
        sources=sources,
        response_time_seconds=response_time_seconds,
        retrieval_latency_seconds=retrieval["retrieval_latency_seconds"],
        use_deepeval=config.use_deepeval,
    )

    save_qa_record(
        config=config,
        question=question,
        answer=answer,
        sources=sources,
        metrics=metrics,
        response_time_seconds=response_time_seconds,
        retrieval_latency_seconds=retrieval["retrieval_latency_seconds"],
    )
    logger.info("Answered question in %.3f seconds", response_time_seconds)

    return {
        "answer": answer,
        "sources": sources,
        "metrics": metrics,
        "response_time_seconds": response_time_seconds,
        "retrieval_latency_seconds": retrieval["retrieval_latency_seconds"],
        "confidence_score": metrics.get("faithfulness", 0.0)
        * metrics.get("retrieval_quality", 0.0),
    }
