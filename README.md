# 🤖 Solvex Telegram RAG Bot

A Telegram support bot for the Solvex platform that uses **Retrieval-Augmented Generation (RAG)** to answer user questions from your documentation, powered by Sentence Transformers and Claude.

---

## Features

- **Hybrid RAG** — combines document retrieval with Claude's reasoning for accurate, grounded answers
- **Smart chunking** — sentence-aware chunking with overlap to preserve context at boundaries
- **Semantic + keyword retrieval** — reranks candidates using a blend of embedding similarity and keyword overlap
- **Conversation memory** — maintains per-user chat history across turns (configurable window)
- **FAQ inline menu** — one-tap answers to common questions via Telegram inline keyboard
- **Greeting detection** — handles greetings and small talk without hitting the AI pipeline

---

## Prerequisites

- Python 3.9+
- A Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- An Anthropic API Key
- A plain-text knowledge base file (`solvex.txt`)

---

## Installation

```bash
git clone <your-repo-url>
cd <repo-directory>
pip install python-telegram-bot anthropic sentence-transformers numpy
```

---

## Configuration

Create a `config.py` file in the root directory and add your credentials:

```python
import os

os.environ["TELEGRAM_BOT_TOKEN"] = "your-telegram-bot-token"
os.environ["ANTHROPIC_API_KEY"]  = "your-anthropic-api-key"
```

> `config.py` is imported automatically before the bot starts — do not commit this file to version control. Add it to `.gitignore`.

---

## Project Structure

```
.
├── bot.py          # Core RAG bot logic (chunking, retrieval, Claude integration)
├── run.py          # Entry point — loads config then starts the bot
├── config.py       # Your API keys (not committed to git)
├── solvex.txt      # Knowledge base document
└── README.md
```

---

## Usage

Add your tokens to `config.py`, then run:

```bash
python run.py
```

The bot will load the document, generate embeddings, and begin polling for messages.

---

## Bot Commands

| Command  | Description                            |
|----------|----------------------------------------|
| `/start` | Welcome message + FAQ menu             |
| `/menu`  | Show the FAQ inline keyboard           |
| `/help`  | Usage instructions and examples        |
| `/info`  | Bot stats (chunks, model, memory size) |
| `/clear` | Reset your conversation memory         |
| `/stop`  | End the session and clear memory       |

---

## How It Works

1. **Document loading** — `solvex.txt` is read and split into overlapping chunks using sentence-aware boundaries.
2. **Embedding** — All chunks are encoded with `all-mpnet-base-v2` (Sentence Transformers) at startup.
3. **Retrieval** — On each query, the top candidates are retrieved by cosine similarity, then reranked using a 70/30 blend of semantic score and keyword overlap.
4. **Generation** — The top chunks are injected into a prompt and sent to Claude, which generates a concise, grounded answer.
5. **Memory** — Each user's last 6 messages (3 exchanges) are kept and passed as conversation history to maintain context.

---

## Configuration Constants

These can be adjusted at the top of `bot.py`:

| Constant            | Default | Description                              |
|---------------------|---------|------------------------------------------|
| `TOP_K`             | `5`     | Number of chunks retrieved per query     |
| `MAX_CONTEXT_CHARS` | `3000`  | Max characters of context sent to Claude |
| `CHUNK_SIZE`        | `500`   | Target characters per chunk              |
| `CHUNK_OVERLAP`     | `100`   | Overlap characters between chunks        |

---

## Notes

- The bot uses `claude-3-haiku-20240307` for fast, cost-efficient responses. You can swap this for any Claude model in `answer_with_rag()`.
- If `config.py` is not found, the bot falls back to environment variables, then prompts for credentials at startup.
- The knowledge base (`solvex.txt`) must be present at startup or the bot will exit.
