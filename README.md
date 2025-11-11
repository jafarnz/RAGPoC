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
The agent expects two JSON files alongside `rag.py`:
- `categories.json`: the nested category tree used for embeddings.
- `category_data.json`: mutable quantities per category (pre-populated with sample data).
If you need a clean slate, delete `category_data.json` before the first run and the script will recreate it as quantities are set.

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
You will be prompted with `Query:`. Type natural-language requests such as “update UV Protection to 120” or “show Serum stock”. The agent prints tool calls in the console followed by the final response. Use `Ctrl+C` to exit.

## Notes
- The first run computes embeddings for every category; keep the session alive to reuse the in-memory FAISS index.
- `category_data.json` is updated in place whenever `set_qty` is called, so commit or back up this file if you care about the history.
