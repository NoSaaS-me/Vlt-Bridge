"""Entry point for running the FastAPI application."""

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Read port from environment variable, default to 8000 for FastAPI server
    # (matches frontend proxy config and development scripts)
    # Can be overridden: PORT=7860 python main.py
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
