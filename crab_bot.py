#!/usr/bin/env python3
"""
Improved Telegram RAG Bot (Sentence Transformers + Claude)

Enhancements:
- Hybrid RAG (combines document context + AI knowledge)
- Smart chunking with overlap
- Better embedding model
- Improved retrieval with reranking
- Claude Sonnet for better reasoning
- Config file integration
"""

import os
import sys
import asyncio
import logging
import re
import numpy as np

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import anthropic
from sentence_transformers import SentenceTransformer

# ================= IMPORT CONFIG =================
try:
    import config  # This loads your API keys into environment
    logger = logging.getLogger(__name__)
    logger.info("✅ Config loaded successfully")
except ImportError:
    print("⚠️  config.py not found - will use environment variables or prompt for keys")

# ================= CONFIG =================

DOCUMENT_PATH = "solvex.txt"
TOP_K = 5  # Increased from 3 for better retrieval
MAX_CONTEXT_CHARS = 3000  # Increased for more comprehensive context
CHUNK_SIZE = 500  # Optimal chunk size
CHUNK_OVERLAP = 100  # Overlap to preserve context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= FAQ MENU =================

FAQ_QUESTIONS = {
    "ai_models": {
        "question": "What AI models can I use?",
        "answer": "Any model that can be called via API. OpenAI, Claude, Llama, Mistral, custom fine-tuned models - if it can take input and produce output, it works on Solvex."
    },
    "payments": {
        "question": "How do payments work?",
        "answer": "Buyers pay in SLVX when ordering. Funds are held in escrow. When the job is marked complete, payment is released instantly to your connected wallet."
    },
    "fees": {
        "question": "What are the fees?",
        "answer": "5% platform fee on completed transactions. Solana network fees are negligible (fractions of a cent). No monthly fees, no setup costs."
    },
    "safety": {
        "question": "Is my AI safe?",
        "answer": "Your AI runs on your infrastructure. We never access your models. Authentication is via a unique secret key that only you possess."
    },
    "getting_started": {
        "question": "How do I get started?",
        "answer": "Create an account, get your secret key, connect your AI, list a service, and start earning. The whole process takes under 10 minutes."
    }
}

def create_faq_menu():
    """Create inline keyboard with FAQ buttons"""
    keyboard = [
        [
            InlineKeyboardButton("🤖 AI Models", callback_data="faq_ai_models"),
            InlineKeyboardButton("💰 Payments", callback_data="faq_payments")
        ],
        [
            InlineKeyboardButton("💳 Fees", callback_data="faq_fees"),
            InlineKeyboardButton("🔒 AI Safety", callback_data="faq_safety")
        ],
        [
            InlineKeyboardButton("🚀 Getting Started", callback_data="faq_getting_started")
        ],
        [
            InlineKeyboardButton("❓ Ask Custom Question", callback_data="ask_custom")
        ],
        [
            InlineKeyboardButton("❌ Stop Bot", callback_data="stop_bot")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# ================= LOAD DOCUMENT =================

if not os.path.exists(DOCUMENT_PATH):
    print(f"❌ Missing document: {DOCUMENT_PATH}")
    sys.exit(1)

with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    DOCUMENT_TEXT = f.read()

logger.info(f"📄 Loaded document: {len(DOCUMENT_TEXT)} characters")

# ================= IMPROVED CHUNKING =================

def smart_chunking(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Smart chunking with overlap to preserve context across boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk size
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            
            # Create overlap by keeping last part of current chunk
            words = current.split()
            if len(words) > 20:  # Only create overlap if chunk is substantial
                overlap_text = ' '.join(words[-20:])  # Keep last ~20 words
                current = overlap_text + " " + sentence
            else:
                current = sentence
        else:
            current += " " + sentence if current else sentence
    
    # Add the last chunk
    if current.strip():
        chunks.append(current.strip())
    
    return chunks

DOCUMENT_CHUNKS = smart_chunking(DOCUMENT_TEXT)
logger.info(f"✂️  Created {len(DOCUMENT_CHUNKS)} chunks with overlap")

# ================= BETTER EMBEDDINGS =================

# Upgraded embedding model for better accuracy
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # Better than MiniLM
logger.info("🔄 Loading embedding model...")

CHUNK_EMBEDDINGS = embedding_model.encode(
    DOCUMENT_CHUNKS,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

logger.info("✅ Embeddings created")

# ================= IMPROVED RETRIEVAL =================

def retrieve_chunks_semantic(query: str, k: int = TOP_K):
    """
    Retrieve semantically similar chunks with keyword boosting.
    
    Args:
        query: User's question
        k: Number of top chunks to return
    
    Returns:
        List of most relevant chunks
    """
    # Get semantic embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Calculate semantic similarity
    semantic_scores = np.dot(CHUNK_EMBEDDINGS, query_embedding)
    
    # Get more candidates than needed for reranking
    num_candidates = min(k * 2, len(DOCUMENT_CHUNKS))
    top_indices = np.argsort(semantic_scores)[-num_candidates:][::-1]
    
    # Rerank using keyword overlap
    query_words = set(query.lower().split())
    reranked = []
    
    for idx in top_indices:
        chunk = DOCUMENT_CHUNKS[idx]
        chunk_words = set(chunk.lower().split())
        
        # Calculate keyword overlap score
        keyword_overlap = len(query_words & chunk_words) / max(len(query_words), 1)
        
        # Combine semantic and keyword scores
        combined_score = semantic_scores[idx] * 0.7 + keyword_overlap * 0.3
        
        reranked.append((chunk, combined_score, idx))
    
    # Sort by combined score and return top k
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Log retrieval info
    logger.info(f"🔍 Retrieved {k} chunks for query: {query[:50]}...")
    
    return [chunk for chunk, score, idx in reranked[:k]]

# ================= HYBRID RAG ANSWER =================

def answer_with_rag(question: str, claude_client: anthropic.Anthropic, conversation_history: list = None) -> str:
    """
    Generate answer using hybrid RAG approach.
    Combines document context with AI knowledge for better responses.
    
    Args:
        question: User's question
        claude_client: Anthropic client instance
    
    Returns:
        Answer string
    """
    retrieved_chunks = retrieve_chunks_semantic(question, TOP_K)

    if not retrieved_chunks:
        return (
            "I couldn't find relevant information in the provided document. "
            "Could you rephrase your question?"
        )

    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        if len(context) + len(chunk) <= MAX_CONTEXT_CHARS:
            context += f"[Context {i}]\n{chunk}\n\n"
        else:
            break

    # Hybrid RAG prompt - allows both document context and general knowledge
    prompt = f"""You are an expert product support assistant helping customers with their questions.

**Your Primary Task:** Answer the customer's question using the documentation provided below as your PRIMARY and most authoritative source.

**Response Style:** Keep answers SHORT and CONCISE (2-4 sentences max). Get straight to the point.

**Guidelines:**
1. **Respond to normal text normally:** like greetings, small talk, or casual conversation.

2. **Be Brief:** Don't over-explain. Answer directly and concisely.
   - Clearly indicate: "According to the documentation: ..." for doc-based info
   - Use "Generally..." for general knowledge only when needed
   
3. **Be Honest:** If the specific answer isn't in the documentation, say: "I don't have that specific information in our documentation."

**Documentation Context:**
{context}

**Customer Question:**
{question}

**Your Answer (keep it SHORT):**"""

    try:
        # Build message history for Claude
        messages = []
        
        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current question with context
        messages.append({"role": "user", "content": prompt})
        
        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",  # Upgraded from Haiku for better reasoning
            max_tokens=300,  # Reduced for shorter, concise answers
            temperature=0.1,  # Low temperature for factual accuracy
            messages=messages
        )
        
        answer = response.content[0].text.strip()
        logger.info(f"✅ Generated answer ({len(answer)} chars)")
        return answer
        
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise

# ================= GREETING DETECTION =================

def is_greeting(text: str) -> bool:
    """
    Detect if message is a greeting or casual conversation.
    Returns True if it's a greeting/small talk, False if it's a real question.
    """
    text_lower = text.lower().strip()
    
    # Common greetings
    greetings = [
        'hi', 'hello', 'hey', 'hola', 'greetings', 'good morning', 
        'good afternoon', 'good evening', 'howdy', 'sup', "what's up",
        'whats up', 'yo', 'hiya'
    ]
    
    # Casual responses
    casual = [
        'thanks', 'thank you', 'thx', 'ty', 'appreciate it', 'cool',
        'ok', 'okay', 'alright', 'got it', 'understood', 'nice',
        'great', 'awesome', 'perfect', 'bye', 'goodbye', 'see you',
        'later', 'cheers', 'ciao'
    ]
    
    # Check if the entire message is just a greeting
    if text_lower in greetings + casual:
        return True
    
    # Check if message starts with greeting (but might have more)
    for greeting in greetings:
        if text_lower.startswith(greeting) and len(text.split()) <= 3:
            return True
    
    # Check for question marks - likely a real question
    if '?' in text:
        return False
    
    return False

def get_greeting_response(text: str) -> str:
    """Generate appropriate response to greetings"""
    text_lower = text.lower().strip()
    
    # Goodbyes
    if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'later', 'ciao']):
        return "👋 Goodbye! Feel free to come back anytime if you have questions!"
    
    # Thanks
    if any(word in text_lower for word in ['thanks', 'thank you', 'thx', 'appreciate']):
        return "You're welcome! Happy to help. Let me know if you have any other questions! 😊"
    
    # Positive feedback
    if any(word in text_lower for word in ['cool', 'nice', 'great', 'awesome', 'perfect', 'got it', 'understood']):
        return "Glad I could help! Anything else you'd like to know?"
    
    # Standard greetings
    return "👋 Hello! I'm here to help you with questions about our product and services. What would you like to know?"

# ================= TELEGRAM BOT =================

class TelegramRAGBot:
    def __init__(self, token: str, claude_key: str):
        self.bot = Bot(token=token)
        self.claude = anthropic.Anthropic(api_key=claude_key)
        self.update_id = 0
        self.running = True
        self.conversation_memory = {}  # Store conversation history per user
        self.max_memory_messages = 6  # Keep last 6 messages (3 exchanges)
        logger.info("🤖 Bot initialized with conversation memory")

    async def send_welcome(self, chat_id: int):
        await self.bot.send_message(
            chat_id=chat_id,
            text=(
                "👋 Welcome to Solvex Support Bot!\n\n"
                "💡 Ask me anything about Solvex - our platform, features, or how to get started.\n\n"
                "📋 **Quick answers to common questions:**\n"
                "Tap any button below for instant answers!"
            ),
            reply_markup=create_faq_menu()
        )
    
    def get_conversation_history(self, chat_id: int) -> list:
        """Get conversation history for a specific user"""
        return self.conversation_memory.get(chat_id, [])
    
    def add_to_conversation(self, chat_id: int, role: str, content: str):
        """Add a message to conversation history"""
        if chat_id not in self.conversation_memory:
            self.conversation_memory[chat_id] = []
        
        self.conversation_memory[chat_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only recent messages to avoid token limits
        if len(self.conversation_memory[chat_id]) > self.max_memory_messages:
            self.conversation_memory[chat_id] = self.conversation_memory[chat_id][-self.max_memory_messages:]
    
    def clear_conversation(self, chat_id: int):
        """Clear conversation history for a user"""
        if chat_id in self.conversation_memory:
            del self.conversation_memory[chat_id]
            logger.info(f"🗑️ Cleared conversation history for user {chat_id}")

    async def handle_command(self, chat_id: int, command: str):
        if command == "/start":
            await self.send_welcome(chat_id)
        elif command == "/menu":
            await self.bot.send_message(
                chat_id=chat_id,
                text="📋 **Common Questions:**\n\nTap any button for instant answers:",
                reply_markup=create_faq_menu()
            )
        elif command == "/stop":
            # Clear conversation memory
            self.clear_conversation(chat_id)
            
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    "👋 **Thank you for using Solvex Support Bot!**\n\n"
                    "Your conversation has been ended and memory cleared.\n\n"
                    "Feel free to return anytime with /start if you have more questions!\n\n"
                    "Have a great day! 😊"
                )
            )
            logger.info(f"🛑 User {chat_id} stopped the bot via command")
        elif command == "/help":
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    "❓ **How to use this bot:**\n\n"
                    "Simply type your question about Solvex and I'll answer based on our documentation.\n\n"
                    "**Quick Access:**\n"
                    "/menu - Show FAQ buttons\n"
                    "/stop - End conversation\n"
                    "/clear - Reset conversation memory\n"
                    "/info - Bot statistics\n\n"
                    "**Examples:**\n"
                    "• What AI models can I use?\n"
                    "• How do payments work?\n"
                    "• What are the fees?\n\n"
                    "I'm here to help! 🚀"
                )
            )
        elif command == "/info":
            memory_count = len(self.conversation_memory.get(chat_id, []))
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"📊 **Bot Information:**\n\n"
                    f"📄 Document chunks: {len(DOCUMENT_CHUNKS)}\n"
                    f"🔍 Retrieval strategy: Hybrid semantic + keyword\n"
                    f"🤖 AI Model: Claude 3.5 Sonnet\n"
                    f"🎯 Approach: Hybrid RAG (Doc + General Knowledge)\n"
                    f"💬 Messages in memory: {memory_count}\n\n"
                    f"Version: 2.0 (Improved with Memory)"
                )
            )
        elif command == "/clear":
            self.clear_conversation(chat_id)
            await self.bot.send_message(
                chat_id=chat_id,
                text="🗑️ Conversation memory cleared! Starting fresh."
            )
        else:
            await self.bot.send_message(
                chat_id=chat_id, 
                text="❓ Unknown command. Try /help or /menu"
            )

    async def process_message(self, message):
        chat_id = message.chat.id
        text = message.text.strip()

        # Handle commands
        if text.startswith("/"):
            await self.handle_command(chat_id, text)
            return

        logger.info(f"❓ Message from {chat_id}: {text}")

        # Check if it's a greeting or casual conversation
        if is_greeting(text):
            response = get_greeting_response(text)
            await self.bot.send_message(chat_id=chat_id, text=response)
            logger.info(f"👋 Sent greeting response to {chat_id}")
            return

        # Send typing indicator for real questions
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass

        # Generate answer with conversation history
        try:
            # Get conversation history for context
            history = self.get_conversation_history(chat_id)
            
            # Generate answer
            answer = answer_with_rag(text, self.claude, history)
            
            # Add user question and bot answer to memory
            self.add_to_conversation(chat_id, "user", text)
            self.add_to_conversation(chat_id, "assistant", answer)
            
            await self.bot.send_message(chat_id=chat_id, text=answer)
            logger.info(f"✅ Answered user {chat_id}")
            
        except Exception as e:
            logger.exception("❌ Error generating answer")
            await self.bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ I encountered an error processing your question. "
                    "Please try rephrasing or contact support if the issue persists."
                )
            )
    
    async def handle_callback_query(self, callback_query):
        """Handle FAQ button clicks"""
        try:
            query_data = callback_query.data
            chat_id = callback_query.message.chat.id
            
            logger.info(f"📲 Callback received: '{query_data}' from chat {chat_id}")
            
            # Answer the callback query FIRST to remove loading state
            try:
                await self.bot.answer_callback_query(callback_query.id)
                logger.info(f"✅ Answered callback query")
            except Exception as e:
                logger.error(f"❌ Error answering callback: {e}")
            
            # Handle stop bot request
            if query_data == "stop_bot":
                self.clear_conversation(chat_id)
                
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "👋 Thank you for using Solvex Support Bot!\n\n"
                        "Your conversation has been ended and memory cleared.\n\n"
                        "Feel free to return anytime with /start if you have more questions!\n\n"
                        "Have a great day! 😊"
                    )
                )
                logger.info(f"🛑 User {chat_id} stopped the bot")
                return
            
            # Handle ask custom question
            elif query_data == "ask_custom":
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "💬 Ask me anything!\n\n"
                        "Just type your question and I'll search our documentation "
                        "to give you the best answer.\n\n"
                        "Examples:\n"
                        "• How do I integrate my API?\n"
                        "• What programming languages are supported?\n"
                        "• Can I customize my AI responses?"
                    )
                )
                logger.info(f"💬 User {chat_id} wants to ask custom question")
                return
            
            # Handle FAQ queries
            elif query_data.startswith("faq_"):
                faq_key = query_data.replace("faq_", "")
                logger.info(f"🔍 Looking for FAQ key: '{faq_key}'")
                logger.info(f"Available FAQ keys: {list(FAQ_QUESTIONS.keys())}")
                
                if faq_key in FAQ_QUESTIONS:
                    faq = FAQ_QUESTIONS[faq_key]
                    logger.info(f"✅ Found FAQ: {faq['question']}")
                    
                    # Send the answer - SIMPLE TEXT, NO MARKDOWN
                    answer_text = f"❓ {faq['question']}\n\n{faq['answer']}\n\n💬 Have more questions? Choose an option below:"
                    
                    try:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text=answer_text
                        )
                        logger.info(f"✅ Sent FAQ answer to {chat_id}")
                    except Exception as e:
                        logger.error(f"❌ Error sending FAQ answer: {e}")
                        # Try without emojis as fallback
                        simple_text = f"{faq['question']}\n\n{faq['answer']}"
                        await self.bot.send_message(chat_id=chat_id, text=simple_text)
                    
                    # Show menu again
                    try:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text="📋 What would you like to know next?",
                            reply_markup=create_faq_menu()
                        )
                        logger.info(f"✅ Sent FAQ menu to {chat_id}")
                    except Exception as e:
                        logger.error(f"❌ Error sending FAQ menu: {e}")
                else:
                    logger.error(f"❌ FAQ key '{faq_key}' not found!")
                    logger.error(f"Received: '{faq_key}', Available: {list(FAQ_QUESTIONS.keys())}")
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text="Sorry, I couldn't find that answer. Please try /menu again."
                    )
            else:
                logger.warning(f"⚠️ Unknown callback data: {query_data}")
                await self.bot.send_message(
                    chat_id=chat_id,
                    text="Unknown action. Please try /menu"
                )
                
        except Exception as e:
            logger.exception(f"❌ CRITICAL Error in handle_callback_query: {e}")
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text="An error occurred. Please try /menu again."
                )
            except:
                logger.error("❌ Could not send error message to user")
                pass

    async def poll(self):
        logger.info("🚀 Bot is running and waiting for messages...")

        while self.running:
            try:
                updates = await self.bot.get_updates(
                    offset=self.update_id + 1,
                    timeout=60
                )

                for update in updates:
                    self.update_id = update.update_id
                    
                    logger.info(f"📨 Received update ID: {update.update_id}")

                    # Handle regular text messages
                    if update.message and update.message.text:
                        logger.info(f"💬 Processing text message: {update.message.text[:50]}")
                        await self.process_message(update.message)
                    
                    # Handle button clicks (callback queries)
                    elif update.callback_query:
                        logger.info(f"🔘 Processing callback query: {update.callback_query.data}")
                        await self.handle_callback_query(update.callback_query)
                    
                    else:
                        logger.warning(f"⚠️ Unknown update type: {update}")

            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.exception("❌ Polling error")
                await asyncio.sleep(5)

    async def start(self):
        try:
            me = await self.bot.get_me()
            logger.info(f"✅ Connected as @{me.username}")
            await self.poll()
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            raise

# ================= MAIN =================

def main():
    """
    Main entry point - loads credentials and starts the bot.
    Priority: config.py > environment variables > user input
    """
    # Try to get credentials from environment (set by config.py or system)
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    # Fallback to user input if not found
    if not telegram_token:
        telegram_token = input("Telegram Bot Token: ").strip()
    if not claude_key:
        claude_key = input("Claude API Key: ").strip()

    # Validate credentials
    if not telegram_token or not claude_key:
        print("❌ Missing credentials. Please set them in config.py or environment variables.")
        return

    logger.info("🔑 Credentials loaded successfully")

    # Initialize and start bot
    bot = TelegramRAGBot(telegram_token, claude_key)

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()