# PoCRAG Quick Setup

## Requirements
- Python 3.10+ (with pip)
- Google AI Studio API key that can access Gemini 2.5 Flash + `text-embedding-004`
- (Optional) `python -m venv` for an isolated environment

## Install Dependencies
```bash
cd /path/to/PoCRAG
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install --upgrade pip
pip install google-generativeai faiss-cpu numpy
```

## Configure Data Files
The agent expects a single `categories.json` file alongside `rag.py`. It stores the nested tree (e.g., `Makeup > Eyes > Mascaras`) with a `_meta` block at every node that holds its random identifier, optional historic quantity, and definition text.

## Provide Your API Key
`rag.py` currently configures Gemini with a placeholder:
```python
genai.configure(api_key="API_KEY")
```
Replace `API_KEY` with your real key **or** change the line to read from an environment variable, e.g.:
```python
import os
...
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
```
Then export the key before running:
```bash
export GOOGLE_API_KEY="paste-your-key"
```

## Run the Agent
```bash
python rag.py
```
You will be prompted with `Query:`. Type natural-language requests such as “eye lines”, “show Eyes”, or “UV protection”. The agent fuzzy-matches the best category and returns its identifier (and, for parent nodes, the covered children). The console shows tool calls followed by the final response. Use `Ctrl+C` to exit.

## Notes
- The first run computes embeddings for every category; keep the session alive to reuse the in-memory FAISS index.
- `categories.json` already contains identifiers and legacy quantity values. Edit it directly if you need to adjust the catalog.

## jafarniaz 2025
