import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from config import Config
from logger import logger
from datetime import datetime
import re
import os
from vector_store import build_index, load_existing_index, query_index

config = Config()
os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY

# Create Telegram bot instance
if not config.TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
    exit(1)  # Exit the script if token is missing

# Create Telegram bot instance
try:
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    logger.info("Telegram bot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Telegram bot: {str(e)}")
    exit(1)

# Telegram message handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    print("Received /start command")
    bot.reply_to(message, "Welcome to the Telegram Bot! Type a message to get started or send /help for available commands.")

@bot.message_handler(commands=['help'])
def send_help(message):
    print("Received /help command")
    bot.reply_to(message, """
Here are the available commands:
/start - Start the bot
/help - Show this help message
    
You can also just type a message and I'll respond based on my programming!
""")

########################################################################################
@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    """
    Handles incoming document messages. This function checks if the uploaded document is a PDF,
    downloads it, saves it to the local filesystem, and indexes it for later retrieval.

    Args:
        message (Message): The incoming message containing the document.

    Returns:
        None
    """
    # Check if the uploaded document is a PDF
    if not message.document.mime_type == 'application/pdf':
        bot.reply_to(message, "Please upload a PDF document.")
        return

    try:
        # Get file information and download the file
        logger.info(f"Downloading file with file_id:{message.document.file_id}")
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error downloading your document.")
        return

    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    pdf_path = os.path.join("uploads", message.document.file_name)
    
    try:
        # Save the downloaded file to the uploads directory
        with open(pdf_path, 'wb') as new_file:
            new_file.write(downloaded_file)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error saving your document.")
        return
    
    # Ensure the chroma_db directory exists
    os.makedirs("chroma_db", exist_ok=True)

    try:
        # Build the index for the uploaded PDF
        build_index(pdf_path, persist_directory="chroma_db")
        bot.reply_to(message, "PDF uploaded and indexed successfully.")
    except Exception as e:
        logger.error(f"Error indexing PDF: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error indexing your document.")
########################################################################################

@bot.message_handler(commands=['query'])
def handle_query(message):
    """
    Handles the /query command from a Telegram message.
    This function extracts the query from the message, processes it using a vector database, processing consists of
    converting the query to a vector, searching the vector database for similar vectors, and retrieving the most
    relevant result, sends the result along with query as context to underlying LLM and returns the reponse from the LLM. 
    If the query is empty or an error occurs during processing, an appropriate error message is sent back to the user.
    Args:
        message (telebot.types.Message): The Telegram message object containing the /query command.
    Returns:
        None
    """
    try:
        query = message.text[len('/query '):].strip()
        if not query:
            bot.reply_to(message, "Please provide a query after the /query command.")
            return
        
        logger.info(f"Received query: {query}")
        vectordb = load_existing_index(persist_directory="chroma_db")
        result = query_index(vectordb, query)
        bot.reply_to(message, result)
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your query.")
########################################################################################

# Message handler class
class MessageHandler:
    def __init__(self):
        # Store user session data (could be moved to a database for persistence)
        self.sessions = config.USER_SESSIONS
    
    def _get_user_session(self, user_id):
        """Get or create a new session for the user"""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'last_interaction': datetime.now(),
                'conversation': [],
                'context': {}
            }
        return self.sessions[user_id]
    
    def _update_session(self, user_id, message, response):
        """Update the user session with new interaction"""
        session = self._get_user_session(user_id)
        session['last_interaction'] = datetime.now()
        session['conversation'].append({
            'user': message,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        # Limit conversation history (optional)
        if len(session['conversation']) > 10:
            session['conversation'] = session['conversation'][-10:]
    
    def process_message(self, user_id, message):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        # Get user session
        session = self._get_user_session(user_id)
        
        # Convert message to lowercase for easier matching
        message_lower = message.lower().strip()
        
        # Basic response logic
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"
        
        elif "help" in message_lower:
            response = ("Here's what I can do:\n"
                      "- Answer questions\n"
                      "- Provide information\n"
                      "- Set reminders\n"
                      "- Tell jokes\n"
                      "Just let me know what you need!")
        
        elif "joke" in message_lower:
            response = "Why don't scientists trust atoms? Because they make up everything! 😄"
        
        elif "time" in message_lower:
            current_time = datetime.now().strftime("%H:%M:%S")
            response = f"The current time is {current_time}"
        
        elif "reminder" in message_lower or "remind" in message_lower:
            # Simple reminder extraction (very basic)
            match = re.search(r"remind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+at\s+|$)", message_lower)
            if match:
                reminder_text = match.group(1)
                response = f"I'll remind you to: {reminder_text}\n(Note: This is a demo - actual reminder functionality would require additional implementation)"
            else:
                response = "What would you like me to remind you about?"
        
        # Example of checking conversation history
        elif "last message" in message_lower:
            if len(session['conversation']) > 0:
                last_message = session['conversation'][-1]['user']
                response = f"Your last message was: '{last_message}'"
            else:
                response = "You haven't sent any previous messages."
        
        # Default response for unknown inputs
        else:
            response = "I'm not sure how to respond to that. Type 'help' to see what I can do."
        
        # Update session with this interaction
        self._update_session(user_id, message, response)        
        return response

# Initialize message handler
handler = MessageHandler()

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    try:
        user_id = message.from_user.id
        incoming_msg = message.text
        
        logger.info(f"Received message from {user_id}: {incoming_msg}")
        
        # Process the message
        response_text = handler.process_message(user_id, incoming_msg)
        
        # Send response
        bot.reply_to(message, response_text)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your request.")

def start_bot():
    bot.infinity_polling()        
