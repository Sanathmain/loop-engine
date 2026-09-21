# Loop Engine

Minimal Writer/Critic multi-agent reasoning loop. A user submits a problem, a Writer proposes a solution, a Critic attacks it, and the Writer revises — up to three rounds — using a provider-independent model interface.

This repo currently uses **MockModel** so the orchestration can be tested without calling a real LLM. OpenAI, Anthropic, or Gemini adapters can be added later by implementing `AIModel` without changing Writer, Critic, or the orchestrator.

## Folder structure

```
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   └── routes.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── writer.py
│   │   └── critic.py
│   ├── models/
│   │   ├── base.py
│   │   └── mock_model.py
│   ├── orchestrator/
│   │   └── engine.py
│   ├── schemas/
│   │   ├── session.py
│   │   ├── writer.py
│   │   └── critic.py
│   ├── database/
│   │   ├── db.py
│   │   └── models.py
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── prompts/
│       ├── writer_prompt.py
│       └── critic_prompt.py
├── tests/
│   ├── conftest.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── requirements.txt
├── README.md
└── .env.example
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Database

Loop Engine uses SQLite. Copy `.env.example` to `.env`, then create the file and tables:

```bash
python -m app.database
```

This creates `loop_engine.db` in the project root with a `sessions` table. FastAPI also creates the schema on startup if it is missing, so this step is optional but useful to verify the database before running the API.

Inspect it with:

```bash
sqlite3 loop_engine.db ".schema sessions"
```

## Run the API

```bash
uvicorn app.main:app --reload
```

The server listens on `http://127.0.0.1:8000`. Open that URL for the live Writer/Critic debate UI (session history, stop button, score trend). API docs stay at `http://127.0.0.1:8000/docs`.

The loop now keeps prior-round memory, stops on plateau/regressions/no blocking issues, and ships the **best-scored** round instead of always the last one.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

## Example: create a session

```bash
curl -s -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"problem": "How can we reduce customer waiting time in our restaurant?"}'
```

Note the returned `id`.

## Example: run the Writer/Critic loop

```bash
curl -s -X POST http://127.0.0.1:8000/sessions/<session_id>/run
```

Inspect stored state at any time:

```bash
curl -s http://127.0.0.1:8000/sessions/<session_id>
```

## Tests

```bash
pytest
```

## Use Gemini

1. Create an API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Put it in `.env`:

```
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-pro-preview
```

3. Restart uvicorn, then `POST /sessions/{id}/run` as usual.

To go back to deterministic fixtures, set `MODEL_PROVIDER=mock`.

OpenAI and Anthropic can be added later as extra `AIModel` classes in `app/models/` and a new branch in `app/models/factory.py`. Writer, Critic, and Orchestrator stay unchanged.
