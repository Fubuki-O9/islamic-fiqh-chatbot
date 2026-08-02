# Islamic Fiqh Chatbot

A multilingual **Retrieval-Augmented Generation (RAG)** chatbot that answers Islamic
jurisprudence (Fiqh) questions in **English, Urdu, Roman Urdu, and Arabic**, across the
**Hanafi, Maliki, and Shafi'i** schools plus a General mode — grounded in **verified
Qur'an and Hadith citations**, with confidence scoring and scholar-referral safeguards
so it never fabricates rulings.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-BM25%20%2B%20MiniLM-5F0921)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)

---

## Features

- **Verified citations only** — answers are grounded in a curated dataset and cite
  real Qur'an and Hadith references instead of inventing them.
- **Four languages** — ask in English, Urdu, Roman Urdu, or Arabic; the reply comes
  back in the same language (text and voice).
- **Madhhab-aware** — choose Hanafi, Maliki, Shafi'i, or General Guidance for
  school-specific rulings.
- **Hybrid retrieval** — BM25 lexical search fused with a **fine-tuned multilingual
  MiniLM** embedding model (0.45 / 0.55 weighting).
- **Confidence + safeguards** — low-confidence, sensitive, or unclear questions are
  routed to a "consult a scholar" response rather than a guessed ruling.
- **Voice** — speech-to-text (Whisper) and text-to-speech (edge-tts) in native voices.
- **WhatsApp channel** — the same engine answers via the WhatsApp Cloud API.
- **Query logging** — optional MongoDB Atlas logging (fails silently if unavailable).

---

## Architecture

```
                 ┌────────────────────────────────────────────┐
  Web / WhatsApp │            Flask API  (app.py)             │
  ─────────────▶ │   /ask   /speak   /whatsapp   /verse       │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
                 │           RAG Engine (rag_engine.py)        │
                 │  gates → hybrid retrieve → confidence →     │
                 │  grounded generation → citation stripping   │
                 └───┬───────────────┬────────────────┬────────┘
                     ▼               ▼                ▼
              BM25 + MiniLM     Google Gemini     MongoDB Atlas
              (fine-tuned)      (generation)      (logging, optional)
```

The fine-tuned embedding model is hosted on the Hugging Face Hub and **downloads
automatically on first run** — see below.

---

## Repository structure

```
.
├── backend/                # Flask API + RAG engine
│   ├── app.py              # server & endpoints (/ask, /speak, /whatsapp, /verse)
│   ├── rag_engine.py       # retrieval, gating, confidence, generation
│   ├── voice.py            # Whisper STT + edge-tts TTS
│   ├── db.py               # optional MongoDB query logging
│   ├── data/               # Qur'an, Hadith, and Fiqh datasets
│   ├── requirements.txt
│   └── .env.example        # copy to .env and add your keys
└── islamic_chatbot_frontend/   # static web UI (HTML/CSS/JS)
    ├── index.html
    ├── script.js
    ├── style.css
    ├── config.js           # backend URL
    └── resize.js
```

---

## The fine-tuned model

The multilingual embedding model was fine-tuned on ~10,700 Islamic Q&A pairs
(English / Urdu / Roman Urdu / Arabic) and is published here:

**🤗 [FubukiO9/fiqh-minilm](https://huggingface.co/FubukiO9/fiqh-minilm)**

You do **not** need to download it manually — `rag_engine.py` pulls it from the Hub
automatically the first time the server starts. To use a different model, set
`HF_MODEL` in your `.env`, or place a local copy in `backend/models/islamic_minilm/`.

---

## Quick start

> Full step-by-step instructions (including FFmpeg and WhatsApp setup) are in the
> **Installation Guide**.

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # then edit .env and add your GEMINI_API_KEY
python app.py                      # runs on http://localhost:5000
```

The first run downloads the fine-tuned model from Hugging Face — give it a minute.

### 2. Frontend

Open a second terminal:

```bash
cd islamic_chatbot_frontend
python -m http.server 5500
```

Then open **http://localhost:5500** in your browser.

*(Check that `islamic_chatbot_frontend/config.js` has `BACKEND_URL: "http://localhost:5000"`.)*

---

## Requirements

- Python 3.10–3.12
- A free **Google Gemini API key** ([aistudio.google.com](https://aistudio.google.com))
- **FFmpeg** (for voice)
- Internet connection (model download + each answer)
- *(optional)* MongoDB Atlas URI for query logging
- *(optional)* WhatsApp Cloud API credentials for the WhatsApp channel

---

## Tech stack

| Layer        | Technology                                                        |
| ------------ | ----------------------------------------------------------------- |
| Backend      | Python, Flask, Flask-CORS                                          |
| Retrieval    | BM25 (rank-bm25) + fine-tuned MiniLM (sentence-transformers)       |
| Generation   | Google Gemini                                                     |
| Voice        | OpenAI Whisper (STT), edge-tts (TTS), FFmpeg                       |
| Storage      | MongoDB Atlas (optional)                                          |
| Messaging    | WhatsApp Cloud API                                                |
| Frontend     | HTML, CSS, vanilla JavaScript                                     |

---

## Note on religious content

This project is an educational tool. It answers from a curated, verified dataset and
cites Qur'an and Hadith sources; for sensitive or personal matters it explicitly
advises consulting a qualified scholar rather than issuing a ruling. It is not a
substitute for a qualified mufti.

---

## Acknowledgements

Final Year Project — University of Management and Technology (UMT), Lahore.
Built by Sannan, Uzair Saqib, M. Hamza, and M. Bilal Alvi.
