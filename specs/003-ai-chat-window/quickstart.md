# Quickstart: AI Chat Window

## Prerequisites
1.  **OpenRouter Key**: Get an API key from [openrouter.ai](https://openrouter.ai).
2.  **Environment**: Set `OPENROUTER_API_KEY` in `backend/.env`.

## Testing the Backend
1.  **Start Server**:
    ```bash
    cd backend
    source .venv/bin/activate
    uvicorn src.api.main:app --reload
    ```
2.  **Test Endpoint**:
    ```bash
    curl -X POST http://localhost:8000/api/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Hello", "history": []}'
    ```
    *Note: This will output raw SSE stream data.*

## Testing the Frontend
1.  **Start Client**:
    ```bash
    cd frontend
    npm run dev
    ```
2.  **Open UI**: Go to `http://localhost:5173`.
3.  **Chat**: Click the "Chat" button in the sidebar. Select a persona and send a message.

## Verification
1.  **Check Logs**: After a chat, check `data/vaults/{user}/Chat Logs/` to see the saved Markdown file.
