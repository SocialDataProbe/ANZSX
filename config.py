import os

# The database is read from Turso only. Streamlit copies top-level keys in
# .streamlit/secrets.toml into the environment at start-up.
TURSO_DATABASE_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")
