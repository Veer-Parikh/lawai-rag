import os
import requests
import json
import streamlit as st

BACKEND_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def chat_query(query: str, domain: str = None, top_k: int = 5):
    """
    Sends a chat query to the backend and returns the parsed JSON response.
    """
    endpoint = f"{BACKEND_URL}/chat"
    payload = {
        "query": query,
        "top_k": top_k
    }
    if domain and domain.lower() != "auto":
        payload["domain"] = domain.lower()

    try:
        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=180 # Give sufficient time for LLM generation
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")
        return None
