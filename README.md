# ATLAS-X Backend

AI-native genomics research portal backend.

## Stack
- FastAPI
- PostgreSQL + pgvector
- SQLAlchemy 2
- Pydantic v2
- AlphaGenome API adapter
- Agent-ready service/tool architecture

## 1. Create environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in credentials.

## 2. Run PostgreSQL

Create a database named `atlas_x`.

The application expects PostgreSQL and enables pgvector if the extension is available.

## 3. Start API

```powershell
uvicorn app.main:app --reload
```

Open:
http://127.0.0.1:8000/docs

## Current milestone

This scaffold implements:
- health endpoint
- analysis/session API
- structured analysis state
- AlphaGenome adapter boundary
- visualization specification models
- SQLAlchemy persistence foundation

The AlphaGenome Atlas/API adapter is deliberately isolated so we can wire the exact current SDK methods without coupling the rest of the application to vendor-specific objects.

See `app/services/alphagenome/client.py`.
