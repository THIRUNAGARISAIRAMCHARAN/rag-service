# RAG service (POST /query)

Files: app.py (service), kb.json (your 50 passages), Dockerfile, requirements.txt, test_rag.py (local tests, no network).

Render: New Web Service -> this repo -> Language Docker -> Free. Env vars: AIPIPE_TOKEN (required), MODEL (optional, default gpt-4.1-mini),
EMB_MODEL (optional, default text-embedding-3-small).
Check: https://YOUR-SERVICE.onrender.com/healthz  ->  {"ok":true,"passages":50,"embeddings_ready":true}
Submit the base URL without /query.
