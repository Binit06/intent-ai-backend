
# 🧠 Intent Classification API (FastAPI + Gemini/BERT)

This is a FastAPI-based backend for classifying user prompts into intents such as:
- `qa_complaint`
- `qa_greetings`
- `qa_account`
- `qa_faq`

It uses **BERT** for AI classification with fallback support for a Gemini-2.0.

---

## 📁 Project Structure

```
.
├── app/
│   ├── api/                # Route handlers
│   ├── db/                 # Database models and session
│   ├── services/           # Business logic (intent prediction, ticketing)
│   ├── templates/          # Prompt templates
│   └── main.py             # FastAPI entrypoint
├── create_db.py            # Script to initialize database tables
├── intent_logs.db          # Sample SQLite database file
├── .env                    # Environment variables
├── processed_data_main.csv # Data used for BERT Training
└── README.md
```

---

## 🚀 Getting Started

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. 🗄️ Create or Load the Database

- To create a fresh database and tables:

```bash
python create_db.py
```

- To **explore sample logs and tickets**, use the included file:

### 👉 [`intent_logs.db`](./intent_logs.db) (📎 dowload and load here → [https://sqliteviewer.app](https://sqliteviewer.app))

You can **download and load** the file into [https://sqliteviewer.app](https://sqliteviewer.app) to browse the logs and tickets.

---

### 3. 🔥 Run the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🌐 Frontend

A minimal frontend is available here:

👉 **[Frontend GitHub Repository](https://github.com/Binit06/intent-ai-frontend)**

```bash
git clone https://github.com/Binit06/intent-ai-frontend.git
cd frontend-repo
npm install
npm run dev
```

---

## 🧾 API Example

### `POST /generate-intent`

Request:
```json
{
  "prompt": "I am unable to login into my account."
}
```

Response:
```json
{
  "prompt": "I am unable to login into my account.",
  "ai_response": "qa_account"
}
```

---

## 📘 Logs & Ticketing

### 🧠 `message_logs` Table

- Records every classified user prompt
- Tracks intent, model used, and timestamp
- Useful for analytics and model training

### 🛠️ `tickets` Table

- Stores support requests from gemini (typically from `qa_complaint`)
- Can be expanded to allow agent assignment, ticket resolution, etc.

---

## 🛡️ Environment Variables

Create a `.env` file:

```env
DATABASE_URL=sqlite+aiosqlite:///./intent_logs.db
GEMINI_API_KEY=your_gemini_api_key_here
```
