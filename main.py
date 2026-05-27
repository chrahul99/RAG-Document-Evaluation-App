from pathlib import Path
from typing import Any
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from app.rag_pipeline import answer_question
from evaluation.evaluator import aggregate_metric_averages
from models.providers import provider_setup_message
from utils.config import get_config
from utils.history import initialize_history_db, load_qa_history
from utils.pdf_processing import process_pdf_files
from utils.reset_data import reset_knowledge_base
from utils.vector_store import add_documents_to_vector_store, vector_store_exists


st.set_page_config(
    page_title="Intelligent Document Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main {
                background: #f7f8fb !important;
            }
            section[data-testid="stSidebar"] {
                background: #101827;
            }
            section[data-testid="stSidebar"] * {
                color: #f7fafc;
            }
            .main .block-container {
                padding-top: 2rem;
                max-width: 1180px;
            }
            [data-testid="stAppViewContainer"] .main .block-container,
            [data-testid="stAppViewContainer"] .main .block-container *,
            [data-testid="stVerticalBlock"] *,
            [data-testid="stMarkdownContainer"] *,
            [data-testid="stMetric"],
            [data-testid="stMetric"] * {
                color: #142033 !important;
                -webkit-text-fill-color: #142033 !important;
            }
            [data-testid="stSidebar"] *,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
                color: #f7fafc !important;
                -webkit-text-fill-color: #f7fafc !important;
            }
            [data-testid="stAppViewContainer"] textarea,
            [data-testid="stAppViewContainer"] textarea * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
                background: #272833 !important;
            }
            [data-testid="stAppViewContainer"] button,
            [data-testid="stAppViewContainer"] button * {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
            .metric-card {
                padding: 1rem;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: #ffffff;
            }
            .source-box {
                padding: 1rem;
                border-left: 4px solid #2563eb;
                background: #ffffff;
                border-radius: 6px;
                margin-bottom: 0.75rem;
            }
            .status-ok {
                color: #047857;
                font-weight: 700;
            }
            .status-warn {
                color: #b45309;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def save_uploaded_files(uploaded_files: list[Any], upload_dir: Path) -> list[Path]:
    saved_paths = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        target_path = upload_dir / safe_name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def render_upload_page(config) -> None:
    render_header(
        "Upload Documents",
        "Add PDFs to the local knowledge base. Files are chunked, embedded, and persisted in ChromaDB.",
    )

    setup_message = provider_setup_message(config)
    if setup_message:
        st.error(setup_message)
        st.info(
            "For OpenAI: edit .env and set OPENAI_API_KEY. For Ollama: set LLM_PROVIDER=ollama and run Ollama locally."
        )
        return

    uploaded_files = st.file_uploader(
        "Drag and drop PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs. Large files are processed page by page.",
    )

    col_left, col_right = st.columns([2, 1])
    with col_left:
        if uploaded_files:
            st.subheader("Selected files")
            file_table = pd.DataFrame(
                [
                    {
                        "file_name": file.name,
                        "size_mb": round(file.size / (1024 * 1024), 3),
                        "status": "ready",
                    }
                    for file in uploaded_files
                ]
            )
            st.dataframe(file_table, use_container_width=True, hide_index=True)

            if st.button("Process and index PDFs", type="primary"):
                with st.status("Processing documents", expanded=True) as status:
                    st.write("Saving uploaded files...")
                    saved_paths = save_uploaded_files(uploaded_files, config.upload_dir)

                    st.write("Extracting text and creating semantic chunks...")
                    processing_result = process_pdf_files(saved_paths, config)

                    st.write("Generating embeddings and updating ChromaDB...")
                    vector_result = add_documents_to_vector_store(
                        processing_result["chunks"],
                        config,
                    )

                    status.update(
                        label="Documents indexed successfully",
                        state="complete",
                        expanded=False,
                    )

                st.success("Upload complete. You can ask questions now.")
                metrics = {
                    "Files": len(saved_paths),
                    "Chunks": processing_result["chunk_count"],
                    "Processing time": f"{processing_result['processing_time_seconds']:.2f}s",
                    "Embedding time": f"{vector_result['embedding_time_seconds']:.2f}s",
                }
                st.session_state["last_upload_metrics"] = metrics
                st.dataframe(
                    pd.DataFrame(processing_result["file_summaries"]),
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.info("Drop PDF files above to begin.")

    with col_right:
        st.subheader("Knowledge base")
        uploaded_count = len(list(config.upload_dir.glob("*.pdf")))
        st.metric("Uploaded PDFs", uploaded_count)
        st.metric("Vector DB", "Ready" if vector_store_exists(config) else "Empty")
        if st.button("Reset knowledge base"):
            reset_knowledge_base(config)
            st.session_state.pop("last_answer_result", None)
            st.session_state.pop("last_question", None)
            st.success("Cleared uploaded PDFs, vector database, and history.")
            st.rerun()
        if "last_upload_metrics" in st.session_state:
            for label, value in st.session_state["last_upload_metrics"].items():
                st.metric(label, value)

        sample_files = list(config.sample_dir.glob("*.pdf"))
        if sample_files:
            st.subheader("Sample PDFs")
            for path in sample_files:
                st.caption(path.name)


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Sources")
    for source in sources:
        safe_content = escape(source.get("content", "")[:1200])
        safe_source = escape(str(source.get("source", "Unknown")))
        safe_page = escape(str(source.get("page", "Unknown")))
        st.markdown(
            f"""
            <div class="source-box">
                <strong>{safe_source}</strong>
                &nbsp; Page {safe_page}
                &nbsp; Relevance {float(source.get("score", 0.0)):.3f}
                <p>{safe_content}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_question_page(config) -> None:
    render_header(
        "Ask Questions",
        "Generate grounded answers using retrieved PDF chunks and show the evidence behind each response.",
    )

    setup_message = provider_setup_message(config)
    if setup_message:
        st.error(setup_message)
        st.info(
            "For OpenAI: edit .env and set OPENAI_API_KEY. For Ollama: set LLM_PROVIDER=ollama and run Ollama locally."
        )
        return

    if not vector_store_exists(config):
        st.warning("Upload and process PDFs before asking questions.")
        return

    question = st.text_area(
        "Question",
        placeholder="Example: What are the main implementation steps described in the document?",
        height=110,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Provider", config.llm_provider)
    with col_b:
        st.metric("Top-K retrieval", config.top_k)
    with col_c:
        st.metric("Chunk size", config.chunk_size)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                result = answer_question(question.strip(), config)
            st.subheader("Answer")
            st.write(result["answer"])

            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Confidence",
                f"{result['confidence_score']:.3f}",
            )
            metric_cols[1].metric(
                "Response time",
                f"{result['response_time_seconds']:.2f}s",
            )
            metric_cols[2].metric(
                "Retrieval latency",
                f"{result['retrieval_latency_seconds']:.2f}s",
            )
            metric_cols[3].metric(
                "Faithfulness",
                result["metrics"].get("faithfulness", 0.0),
            )

            render_sources(result["sources"])
            with st.expander("Evaluation metrics"):
                st.json(result["metrics"])
        except Exception as exc:
            st.error(
                "Unable to answer. Check your API key/provider settings and confirm the vector database is populated."
            )
            st.exception(exc)


def render_evaluation_page(config) -> None:
    render_header(
        "Evaluation Dashboard",
        "Track answer relevance, context relevance, faithfulness, hallucination risk, retrieval quality, and latency.",
    )
    records = load_qa_history(config)
    if not records:
        st.info("Ask a question to generate evaluation metrics.")
        return

    averages = aggregate_metric_averages(records)
    cols = st.columns(4)
    dashboard_metrics = [
        ("Answer relevance", "answer_relevance"),
        ("Context relevance", "context_relevance"),
        ("Faithfulness", "faithfulness"),
        ("Hallucination risk", "hallucination_risk"),
    ]
    for index, (label, key) in enumerate(dashboard_metrics):
        cols[index].metric(label, averages.get(key, 0.0))

    metric_rows = []
    for record in reversed(records):
        row = {
            "created_at": record["created_at"],
            "question": record["question"][:80],
        }
        row.update(record["metrics"])
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    st.subheader("Metric trends")
    trend_columns = [
        "answer_relevance",
        "context_relevance",
        "faithfulness",
        "hallucination_risk",
        "retrieval_quality",
    ]
    chart_df = metrics_df[["created_at", *trend_columns]].melt(
        id_vars="created_at",
        var_name="metric",
        value_name="score",
    )
    fig = px.line(
        chart_df,
        x="created_at",
        y="score",
        color="metric",
        markers=True,
        range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resume-ready operational metrics")
    op_cols = st.columns(3)
    op_cols[0].metric(
        "Avg retrieval latency",
        f"{averages.get('retrieval_latency_seconds', 0.0):.3f}s",
    )
    op_cols[1].metric(
        "Avg response time",
        f"{averages.get('response_time_seconds', 0.0):.3f}s",
    )
    op_cols[2].metric("Questions evaluated", len(records))

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_history_page(config) -> None:
    render_header(
        "History",
        "Review prior questions, answers, sources, and evaluation snapshots.",
    )
    records = load_qa_history(config)
    if not records:
        st.info("No history yet.")
        return

    for record in records:
        with st.expander(f"{record['created_at']} - {record['question'][:90]}"):
            st.markdown("**Question**")
            st.write(record["question"])
            st.markdown("**Answer**")
            st.write(record["answer"])
            st.markdown("**Metrics**")
            st.json(record["metrics"])
            render_sources(record["sources"])


def render_sidebar(config) -> str:
    st.sidebar.title("Document Assistant")
    st.sidebar.caption("RAG + evaluation framework")
    page = st.sidebar.radio(
        "Navigation",
        ["Upload Documents", "Ask Questions", "Evaluation Dashboard", "History"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Runtime")
    st.sidebar.write(f"Provider: `{config.llm_provider}`")
    st.sidebar.write(f"Collection: `{config.collection_name}`")
    st.sidebar.write(f"Uploads: `{config.upload_dir}`")
    return page


def main() -> None:
    apply_theme()
    config = get_config()
    initialize_history_db(config)
    page = render_sidebar(config)

    if page == "Upload Documents":
        render_upload_page(config)
    elif page == "Ask Questions":
        render_question_page(config)
    elif page == "Evaluation Dashboard":
        render_evaluation_page(config)
    else:
        render_history_page(config)


if __name__ == "__main__":
    main()
