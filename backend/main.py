"""Main entry point for FastAPI application."""

from src.api.main import app

__all__ = ["app"]


def main():
    """Run the development server (for manual testing)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
