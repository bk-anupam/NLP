import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # This loads the variables from .env

class Config:
    # Telegram Bot Token
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')        
    # For tracking conversation history
    USER_SESSIONS = {}        
    # Optional: API keys for external services
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)