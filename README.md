# CuraLink

CuraLink is an evidence-backed medical research assistant prototype built as a Bun workspaces monorepo.
It combines a React frontend, a Bun/Express API gateway, and a FastAPI AI research service.

## Project structure

- apps/web — React + Vite chat UI
- apps/api — Express API gateway with SSE chat streaming
- apps/ai — FastAPI research and synthesis service

## Local ports

- Web: http://127.0.0.1:5173
- API: http://127.0.0.1:4000
- AI: http://127.0.0.1:8000

---

## Prerequisites

Make sure these are installed first:

- Bun 1.3+
- Node.js 18+
- Python 3.11+

---

## 1. Install dependencies

From the project root:

```powershell
bun install
```

For the Python AI service, activate the virtual environment and install packages if needed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn httpx pydantic
```

---

## 2. Run all three services

Open 3 terminals from the project root.

### Terminal 1 — AI service

```powershell
cd apps/ai
..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Terminal 2 — API service

```powershell
cd apps/api
bun run src/server.ts
```

### Terminal 3 — Web app

```powershell
cd apps/web
bunx vite --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173
```

---

## 3. Optional environment setup

You can configure the API using:

### apps/api/.env.example

- PORT=4000
- AI_SERVICE_URL=http://127.0.0.1:8000
- CORS_ORIGIN=http://localhost:5173
- MONGODB_URI=your_atlas_srv_connection_string
- MONGODB_DIRECT_URI=optional_direct_mongodb_connection_string_from_compass

### apps/ai/.env.example

- PORT=8000
- GROQ_API_KEY=
- OLLAMA_BASE_URL=
- OLLAMA_MODEL=llama3.1:8b

If no MongoDB connection is supplied, the API uses local JSON persistence automatically.
If Atlas works in Compass but fails in the app, use the direct connection string from Compass in MONGODB_DIRECT_URI instead of the SRV form.
If Atlas works in Compass but not in the app, use the direct connection string from Compass in MONGODB_DIRECT_URI instead of the SRV form.

---

## 4. Verify that everything is running

- Web should open successfully on port 5173
- API health: http://127.0.0.1:4000/api/health
- AI health: http://127.0.0.1:8000/health

---

## 5. Build for production

From the root:

```powershell
bun run build
```

---

## Notes

- The app is intended for educational research assistance only.
- It does not replace professional medical advice.
- For hackathon demo flow, start a session with a disease and location, then ask follow-up research questions.
