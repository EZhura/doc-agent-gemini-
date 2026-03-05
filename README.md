# Doc Agent (Gemini + mini-RAG)

Портфолио-проект AI-автоматизатора: агент отвечает на вопросы по документу строго по источнику и умеет извлекать структуру в JSON.

## Features
- Load TXT document
- Mini-RAG retrieval (lexical cosine similarity)
- Answers strictly from retrieved fragments
- `/extract` command returns structured JSON: summary, key_points, entities

## Setup
1) Install dependencies:
```bash
pip install -r requirements.txt
