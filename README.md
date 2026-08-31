# BharatAssist — Starter Codebase

This is a working scaffold for the "Lite" architecture we discussed: **no login/auth**,
**public LLM (Gemini)**, **RAG-grounded assistant**, **basic PII redaction**.
It's tested and runs — you build features on top of this, not from scratch.

## What's already built and working
- Flask app with 3 core pages: Service Search, Document Simplifier, AI Assistant
- SQLite database (swap for Postgres later if you deploy — see roadmap)
- RAG pipeline: ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`)
- Gemini API wrapper (using the current `google-genai` SDK, not the deprecated one)
- Basic PII redaction (Aadhaar, PAN, phone, email patterns) before any text hits the LLM
- Confidence threshold on the assistant — if retrieval isn't confident, it says so instead of guessing
- 5 sample seeded government services (PAN, passport, ration card, income certificate, driving licence)
- GitHub Actions CI (lint + test) already wired up
- IRT (Information Retrieval Time) logging built into the search endpoint, per your evaluation metric

## Setup

```bash
cd bharatassist
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your Gemini API key (free at https://aistudio.google.com/apikey)

python app.py                   # creates the DB on first run
python seed_data.py             # loads sample services + builds RAG index (run once)
python app.py                   # run again to start the server
```

Visit `http://localhost:5000`.

## Roadmap — what to build next, in order

### Week 1 (Days 1–7): make it real
1. **Replace/expand seed data** — the 5 sample services in `seed_data.py` are placeholders.
   Research and add 10–15 real services relevant to your pilot users. Keep the same fields
   (eligibility, documents, steps, fees, processing time, source URL) so nothing else breaks.
2. **Test the simplifier on a real document** — find an actual government notice/legal PDF,
   run it through `/simplify`, and tune `SIMPLIFY_PROMPT` in `utils/llm.py` based on output quality.
3. **Test the assistant end-to-end** — ask it questions covered by your seeded services, confirm
   it grounds answers correctly, and confirm it says "I don't know" for out-of-scope questions.
4. **Push to GitHub, confirm CI runs** — the workflow in `.github/workflows/ci.yml` runs on every push.

### Week 2 (Days 8–14): polish + deploy
5. **Improve the confidence threshold** — the `CONFIDENCE_THRESHOLD = 0.35` in `app.py` is a guess.
   Test it against real queries and tune it based on when the assistant should vs. shouldn't answer.
6. **Add IRT dashboard** — you're already logging `search_logs`; add a small admin page or script
   to compute average IRT from that table, since it's your primary evaluation metric.
7. **Mobile/responsive check** — test on a phone-sized viewport; Bootstrap handles most of this
   already, but check the assistant chat window and upload form specifically.
8. **Deploy to Render** — connect your GitHub repo, set `GEMINI_API_KEY` as an environment
   variable in Render's dashboard, point the build at `requirements.txt` and start command
   `gunicorn app:app` (add `gunicorn` to requirements.txt for production).

### After the pilot (subsequent deliverables)
9. Multilingual support — translate `SIMPLIFY_PROMPT`/`ASSISTANT_PROMPT` instructions, or detect
   input language and ask Gemini to respond in kind (it already supports this reasonably well
   with a small prompt tweak).
10. Document checklist generator — reuses the same RAG index, new prompt template.
11. Login/history — only add if your rubric requires it. If you do, keep LLM calls anonymous
    (don't pass user identity into the prompt) even though the user is logged in — see
    `utils/llm.py` comments for the Ollama fallback stub if you're required to self-host at that point.

## File map
```
app.py                  Flask routes
utils/redact.py          PII redaction (regex-based)
utils/llm.py             Gemini API calls (simplify + assistant)
utils/rag.py             ChromaDB + embeddings
seed_data.py             Sample service data + RAG indexing
templates/                Bootstrap-based HTML pages
.github/workflows/ci.yml  CI: lint + test on every push
tests/test_app.py         Starter smoke tests — expand these as you add features
```

## Known limitations (be upfront about these in your report)
- PII redaction is regex-based, not exhaustive — document this as a v1 limitation, not a solved problem
- Confidence threshold for RAG grounding needs real tuning against pilot data, not just the placeholder value
- No persistence of uploaded documents (by design, for privacy) — if you need audit logs later, add that deliberately with clear disclosure to users
