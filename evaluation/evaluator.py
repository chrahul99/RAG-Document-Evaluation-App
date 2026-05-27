import re
from collections import Counter
from typing import Any


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "are",
        "was",
        "were",
        "you",
        "your",
        "about",
        "into",
        "have",
        "has",
    }
    return {word for word in words if word not in stop_words}


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _unsupported_claim_ratio(answer: str, context: str) -> float:
    sentences = [item.strip() for item in re.split(r"[.!?]+", answer) if item.strip()]
    if not sentences:
        return 0.0

    unsupported = 0
    for sentence in sentences:
        if _overlap_score(sentence, context) < 0.2:
            unsupported += 1
    return unsupported / len(sentences)


def evaluate_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    response_time_seconds: float,
    retrieval_latency_seconds: float,
    use_deepeval: bool = False,
) -> dict[str, float]:
    """Evaluate RAG quality with DeepEval when enabled, plus local fallbacks."""
    context = "\n".join(source.get("content", "") for source in sources)
    source_scores = [float(source.get("score", 0.0)) for source in sources]
    unsupported_ratio = _unsupported_claim_ratio(answer, context)

    answer_relevance = _overlap_score(question, answer)
    context_relevance = _overlap_score(question, context)
    faithfulness = max(0.0, 1.0 - unsupported_ratio)
    hallucination_risk = min(1.0, unsupported_ratio + (0.25 if not sources else 0.0))
    retrieval_quality = sum(source_scores) / len(source_scores) if source_scores else 0.0

    metrics = {
        "answer_relevance": round(answer_relevance, 3),
        "context_relevance": round(context_relevance, 3),
        "faithfulness": round(faithfulness, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "retrieval_quality": round(retrieval_quality, 3),
        "response_time_seconds": round(response_time_seconds, 3),
        "retrieval_latency_seconds": round(retrieval_latency_seconds, 3),
    }

    if use_deepeval:
        metrics.update(_run_deepeval_metrics(question, answer, sources))

    return metrics


def _run_deepeval_metrics(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, float]:
    """Run DeepEval LLM-as-judge metrics when the package and key are available."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase

        retrieval_context = [source.get("content", "") for source in sources]
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )
        metric_objects = {
            "deepeval_answer_relevance": AnswerRelevancyMetric(
                threshold=0.7,
                include_reason=False,
            ),
            "deepeval_context_relevance": ContextualRelevancyMetric(
                threshold=0.7,
                include_reason=False,
            ),
            "deepeval_faithfulness": FaithfulnessMetric(
                threshold=0.7,
                include_reason=False,
            ),
            "deepeval_hallucination_risk": HallucinationMetric(
                threshold=0.3,
                include_reason=False,
            ),
        }

        results = {}
        for name, metric in metric_objects.items():
            metric.measure(test_case)
            results[name] = round(float(metric.score or 0.0), 3)
        return results
    except Exception:
        return {"deepeval_status": 0.0}


def aggregate_metric_averages(records: list[dict[str, Any]]) -> dict[str, float]:
    """Average metrics for the dashboard."""
    metric_names = [
        "answer_relevance",
        "context_relevance",
        "faithfulness",
        "hallucination_risk",
        "retrieval_quality",
        "response_time_seconds",
        "retrieval_latency_seconds",
    ]
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()

    for record in records:
        metrics = record.get("metrics", {})
        for name in metric_names:
            if name in metrics:
                totals[name] += float(metrics[name])
                counts[name] += 1

    return {
        name: round(totals[name] / counts[name], 3)
        for name in metric_names
        if counts[name]
    }
