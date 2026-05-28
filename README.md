# Intelligent Document Assistant with RAG and Evaluation Framework (AI)

A production-style AI application for uploading PDF documents, asking grounded questions, retrieving source evidence, and evaluating RAG answer quality. The project is designed to be resume-ready for AI Engineer and Software Engineer roles.

## Features

- Multi-PDF upload with drag-and-drop support through Streamlit
- PDF text extraction with PyPDF
- Text cleaning and semantic chunking with LangChain
- Persistent ChromaDB vector database
- OpenAI and Ollama provider support
- Retrieval-augmented generation with source-grounded prompts
- Source transparency with file names, page numbers, chunk previews, and relevance scores
- Automatic evaluation metrics for answer relevance, context relevance, faithfulness, hallucination risk, retrieval quality, response time, and retrieval latency
- Dashboard and history views backed by SQLite

## Architecture

```text
project/
|
|-- app/
|   `-- rag_pipeline.py          # Retrieval, generation, evaluation, and persistence flow
|-- data/
|   |-- samples/                 # Sample PDFs for testing
|   `-- uploads/                 # Uploaded PDFs
|-- evaluation/
|   `-- evaluator.py             # Local deterministic RAG quality metrics
|-- models/
|   `-- providers.py             # OpenAI/Ollama chat and embedding model factories
|-- netlify-ui/                  # Static portfolio UI for Netlify
|-- utils/
|   |-- config.py                # Environment and path configuration
|   |-- create_sample_pdf.py     # Generates the included sample PDF
|   |-- history.py               # SQLite history storage
|   |-- logger.py                # File and console logging
|   |-- pdf_processing.py        # PDF extraction, cleaning, and chunking
|   `-- vector_store.py          # ChromaDB persistence and retrieval
|-- vectorstore/                 # Persistent ChromaDB files
|-- .env                         # Local runtime configuration
|-- .env.example                 # Environment template
|-- DEPLOYMENT.md                # Local Ollama and Netlify instructions
|-- main.py                      # Streamlit app entry point
|-- requirements.txt             # Python dependencies
`-- README.md
```

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
copy .env.example .env
```

For OpenAI, set:

```text
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
```

For Ollama, install Ollama locally, pull models, and set:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

To enable DeepEval LLM-as-judge metrics, set `USE_DEEPEVAL=true` and provide the evaluator model credentials required by DeepEval.

4. Generate the sample PDF if it is missing:

```bash
python utils/create_sample_pdf.py
```

5. Run the app:

```bash
streamlit run main.py
```

## How It Works

1. Upload PDFs on the Upload Documents page.
2. The app saves files into `data/uploads`.
3. `pdf_processing.py` extracts page text, cleans it, and splits it into overlapping chunks.
4. `vector_store.py` embeds chunks and persists them in ChromaDB under `vectorstore`.
5. The Ask Questions page retrieves top chunks, passes them into a guarded prompt, and generates an answer.
6. Source chunks, file names, page numbers, and relevance scores are displayed with the answer.
7. `evaluator.py` scores answer quality and latency.
8. `history.py` stores each question, answer, source set, and metric snapshot in SQLite.
9. The Evaluation Dashboard aggregates quality and latency metrics over time.

## Evaluation Metrics

- Answer relevance: lexical overlap between the question and answer
- Context relevance: lexical overlap between the question and retrieved context
- Faithfulness: estimated support of answer sentences in retrieved context
- Hallucination risk: estimated unsupported claim risk
- Retrieval quality: average Chroma relevance score
- Response time: total end-to-end answer time
- Retrieval latency: vector search time

DeepEval is included and can be enabled with `USE_DEEPEVAL=true`. The app also keeps deterministic fallback metrics so the dashboard works without additional evaluator credentials.

## Screenshots

Upload Screen:
<img width="1882" height="905" alt="upload" src="https://github.com/user-attachments/assets/de2bd4e8-f680-4f9f-adfd-d16860fc61d9" />


- Upload Documents page
- Ask Questions page with source chunks
- Evaluation Dashboard
- History page

## Deployment

For the recommended local plus cloud demo setup, see `DEPLOYMENT.md`.

### Local Ollama Deployment

Use this mode for the full working AI app without an OpenAI API key.

1. Install Ollama from `https://ollama.com`.
2. Pull local models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

3. Set `.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

4. Run:

```bash
python -m streamlit run main.py
```

### Netlify Static UI Demo

Netlify cannot run the Streamlit Python backend or Ollama. Use `netlify-ui/` as a public static portfolio demo.

To deploy, drag this folder into Netlify:

```text
netlify-ui
```

Or connect GitHub and set the publish directory to:

```text
project/netlify-ui
```

### Streamlit Community Cloud

1. Push this project to GitHub.
2. Create a new Streamlit app and set `main.py` as the entry point.
3. Add environment variables in Streamlit secrets.
4. Use OpenAI for hosted deployment, because Ollama requires a local model server.

### Docker or VM

1. Install Python 3.10+.
2. Install requirements.
3. Set `.env`.
4. Run `streamlit run main.py --server.port 8501 --server.address 0.0.0.0`.
5. Mount `data/` and `vectorstore/` as persistent volumes.

## Bullet Points

- Built a production-style RAG document assistant with Streamlit, LangChain, OpenAI/Ollama, and ChromaDB for multi-PDF question answering.
- Implemented PDF ingestion, text cleaning, semantic chunking, persistent vector search, source attribution, and grounded answer generation.
- Added an evaluation dashboard tracking answer relevance, context relevance, faithfulness, hallucination risk, retrieval quality, retrieval latency, and response time.
- Designed a modular Python architecture with provider abstraction, SQLite history, local persistence, logging, and error handling.
- Created measurable product metrics for retrieval latency, chunk count, response time, evaluation score averages, and question history.

## Future Improvements

- Add DeepEval or RAGAS LLM-as-judge metrics for deeper semantic evaluation.
- Add authentication and per-user document collections.
- Add hybrid retrieval with BM25 plus vector search.
- Add reranking with a cross-encoder model.
- Add async ingestion jobs for very large PDF batches.
- Add document deletion and collection management.
- Add Dockerfile and CI checks.
