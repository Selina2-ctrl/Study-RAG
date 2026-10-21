# Study RAG

Upload PDFs / Word docs / text notes, then ask questions (with citations), take quizzes, or flip flashcards.

## Run

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your-key                       # PowerShell: $env:GEMINI_API_KEY="your-key"
python app.py
```

Open http://localhost:5057. First run downloads a small embedding model (~90 MB).
Your index is saved in `data/` so you only upload once. Set `GEMINI_MODEL` to change the model.

## How it works

1. Ingest: extract text per PDF page / Word heading section, split into ~180-word overlapping chunks.
2. Embed: `all-MiniLM-L6-v2` runs locally (free, private). Vectors are saved to disk.
3. Retrieve: embed the question, take the top 6 chunks by cosine similarity.
4. Generate: Gemini answers using only those chunks and cites them as [1], [2]...
   Quizzes and flashcards use the same retrieval, optionally focused on a topic you type.
   Scanned PDFs (images) have no text layer and need OCR first.
