"""
Vercel entrypoint for FastAPI application.
This file imports the FastAPI app from backend/api/app.py
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent / 'backend'
sys.path.insert(0, str(backend_dir))

# Import the FastAPI app
from api.app import app

# Export the app for Vercel
__all__ = ['app']
