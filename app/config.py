"""
Config
------
Central place for API keys and environment configuration.

Works in two environments without any code changes needed between
them:
1. Local development - reads from a .env file via python-dotenv.
2. Streamlit Community Cloud (or any host using Streamlit's secrets
   system) - reads from st.secrets (secrets.toml), since cloud hosts
   don't ship a .env file and secrets.toml values aren't
   automatically exposed as environment variables by Streamlit.

The bridge below copies any matching st.secrets keys into
os.environ (only for keys not already set locally), so every other
module in this project can keep using plain os.getenv(...) either
way - nothing downstream needs to know which environment it's in.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Bridge Streamlit secrets -> environment variables only when
# a secrets.toml file exists.

from pathlib import Path

_SECRET_PATHS = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml",
]

_HAS_SECRETS_FILE = any(path.is_file() for path in _SECRET_PATHS)

if _HAS_SECRETS_FILE:
    try:
        import streamlit as st

        for _key in (
            "GROQ_API_KEY",
            "TAVILY_API_KEY",
            "DATABASE_URL",
            "PG_HOST",
            "PG_PORT",
            "PG_DB",
            "PG_USER",
            "PG_PASSWORD",
        ):
            if _key in st.secrets and not os.getenv(_key):
                os.environ[_key] = str(st.secrets[_key])

    except Exception:
        pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"

# --- PostgreSQL (Idea History persistence) ---
DATABASE_URL = os.getenv("DATABASE_URL")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "startup_validator")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")