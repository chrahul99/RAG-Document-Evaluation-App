# Deployment Guide

This project has two deployment modes:

1. Local full AI app on your PC with Ollama
2. Cloud-hosted static UI demo on Netlify

Netlify can host the UI/portfolio demo, but it cannot run the Streamlit Python backend or local Ollama models. The real RAG app runs locally on your computer.

## Option 1: Local Full App With Ollama

Use this mode when you want the app to actually upload PDFs, embed documents, retrieve chunks, and answer questions.

### 1. Install Ollama

Download and install Ollama:

```text
https://ollama.com
```

### 2. Pull the local models

Open PowerShell and run:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Confirm `.env`

Your `.env` should contain:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

You do not need `OPENAI_API_KEY` for Ollama mode.

### 4. Start the Streamlit app

From the project folder:

```powershell
cd "C:\Users\chrah\Documents\New project\project"
.venv\Scripts\activate
python -m streamlit run main.py
```

Open:

```text
http://localhost:8501
```

## Option 2: Netlify Static UI Demo

Use this mode when you want a public portfolio link.

This is a static demo UI. It shows the app design, architecture, metrics, and workflow, but it does not process PDFs or call AI models.

### Deploy From Netlify Drag-and-Drop

1. Go to Netlify.
2. Choose "Add new site".
3. Choose "Deploy manually".
4. Drag this folder into Netlify:

```text
C:\Users\chrah\Documents\New project\project\netlify-ui
```

### Deploy From GitHub

1. Push the project to GitHub.
2. In Netlify, choose "Import from Git".
3. Set publish directory to:

```text
project/netlify-ui
```

4. No build command is needed.

## Recommended Portfolio Setup

- Use Netlify link as your public demo UI.
- Use screenshots or a short video from the local Ollama app to prove the RAG pipeline works.
- In your README, explain that the production AI backend runs locally with Ollama or can run hosted with OpenAI credentials.
