import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_secret(key, default=""):
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


GROQ_API_KEY = _get_secret("GROQ_API_KEY", "")
EMAIL_SENDER = _get_secret("EMAIL_SENDER", "")
EMAIL_PASSWORD = _get_secret("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = _get_secret("EMAIL_RECIPIENT", "")

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
DATA_MODELS = "data/models"

for p in [DATA_RAW, DATA_PROCESSED, DATA_MODELS]:
    os.makedirs(p, exist_ok=True)