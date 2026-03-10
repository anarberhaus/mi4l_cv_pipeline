import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def get_supabase_client() -> Client:
    """Initialize and return a Supabase client using environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables (or .env file).")
        
    return create_client(url, key)
