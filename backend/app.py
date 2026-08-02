# -*- coding: utf-8 -*-
"""
app.py — Flask server for the Islamic Fiqh chatbot.

Endpoints
  GET  /health            liveness + readiness
  POST /ask               web frontend  -> { question, madhab }
  GET  /whatsapp          Meta webhook verification (hub.challenge)
  POST /whatsapp          Meta Cloud API inbound messages (text now; voice next)

Both channels call the same rag_engine.answer_question(), so answers are
identical across web and WhatsApp.
"""
import os, threading, tempfile, re, json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests

# Load environment variables from a .env file in this folder, so keys like
# GEMINI_API_KEY are picked up automatically without needing to set them in the
# terminal each time. Safe no-op if python-dotenv isn't installed or no .env.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

import rag_engine
import db   # optional MongoDB query logging (fails silently if unavailable)

app = Flask(__name__)
CORS(app)  # allow the static frontend (file:// or localhost) to call /ask

# WhatsApp Cloud API config (from env)
WA_VERIFY_TOKEN = os.environ.get("WA_VERIFY_TOKEN", "fiqh-verify")
WA_TOKEN        = os.environ.get("WA_TOKEN", "")          # permanent/temporary access token
WA_PHONE_ID     = os.environ.get("WA_PHONE_NUMBER_ID", "")
WA_API          = "https://graph.facebook.com/v21.0"

VALID_MADHHABS = {"Hanafi", "Shafi'i", "Maliki", "General"}


# --------------------------------------------------------------- random verse
_QURAN_DATA = None
_QURAN_LOCK = threading.Lock()

def _load_quran():
    global _QURAN_DATA
    if _QURAN_DATA is not None:
        return
    import json, os
    path = os.path.join(os.path.dirname(__file__), "data", "quran_trilingual_clean.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # raw is already a list of 6236 verses
    _QURAN_DATA = raw

@app.get("/verse")
def random_verse():
    """Return a random Qur'anic verse with Arabic, English, Urdu, and metadata."""
    with _QURAN_LOCK:
        if _QURAN_DATA is None:
            _load_quran()
    import random
    v = random.choice(_QURAN_DATA)
    return jsonify({
        "arabic":        v.get("arabic", ""),
        "english":       v.get("english", ""),
        "urdu":          v.get("urdu", ""),
        "surah":         v.get("surah_name_en", ""),
        "surah_ar":      v.get("surah_name_ar", ""),
        "chapter":       v.get("chapter"),
        "ayah":          v.get("ayah"),
        "reference":     f"{v.get('surah_name_en', '')} {v.get('chapter')}:{v.get('ayah')}"
    })


# ----- warm up the engine in the background so the first request isn't slow ---
_warm_thread = None
def _warm():
    try:
        rag_engine.init(verbose=True)
    except Exception as e:
        app.logger.error(f"Engine init failed: {e}")

def start_warmup():
    global _warm_thread
    if _warm_thread is None:
        _warm_thread = threading.Thread(target=_warm, daemon=True)
        _warm_thread.start()


# --------------------------------------------------------------- health
@app.get("/health")
def health():
    return jsonify({"status": "ok", "engine_ready": rag_engine._READY})


# --------------------------------------------------------------- hadith text lookup
_HADITH_LOOKUP = None

def _ensure_hadith_lookup():
    global _HADITH_LOOKUP
    if _HADITH_LOOKUP is None:
        path = os.path.join(os.path.dirname(__file__), "data", "sahih_bukhari_muslim.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _HADITH_LOOKUP = {(h["source"], str(h["hadith_no"])): h for h in data}

@app.get("/hadith-text")
def hadith_text():
    source = request.args.get("source", "").strip()
    number = request.args.get("number", "").strip()
    if not source or not number:
        return jsonify({"found": False, "error": "source and number required"}), 400
    _ensure_hadith_lookup()
    h = _HADITH_LOOKUP.get((source, number))
    if h:
        return jsonify({"found": True, "text_ar": h.get("text_ar", ""), "text_en": h.get("text_en", "")})
    return jsonify({"found": False})


# --------------------------------------------------------------- web /ask
@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("text") or "").strip()
    madhhab  = (data.get("madhab") or data.get("madhhab") or "General").strip()
    prev_answer = (data.get("prev_answer") or "").strip()
    if madhhab not in VALID_MADHHABS:
        madhhab = "General"
    if not question:
        return jsonify({"ok": False, "error": "Question is required."}), 400
    try:
        result = rag_engine.answer_question(
            question, None if madhhab == "General" else madhhab, prev_answer=prev_answer or None)
        db.log_query(question, madhhab, result, channel="web")
        return jsonify(result)
    except Exception as e:
        app.logger.exception("answer failed")
        return jsonify({"ok": False, "error": "internal_error", "detail": str(e)}), 500


# --------------------------------------------------------------- text-to-speech
def _detect_tts_lang(text):
    """Pick a spoken language from the text: Urdu-only letters -> ur; other
    Arabic-block -> ar; otherwise en (covers English and Roman Urdu)."""
    t = text or ""
    if re.search(r"[\u067E\u0686\u0698\u06A9\u06AF\u06BA\u06BE\u06C1\u06CC\u06D2\u0679\u0688\u0691]", t):
        return "ur"
    if re.search(r"[\u0600-\u06FF]", t):
        return "ar"
    return "en"


@app.post("/speak")
def speak():
    """Generate speech audio (mp3) for a piece of text using native edge-tts
    voices (English/Urdu/Arabic). Works on any machine — no OS voice needed."""
    import voice  # lazy import so the server still starts if edge-tts isn't installed
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400

    # Clean citations/markdown so they aren't read aloud, then pick a voice.
    clean = voice.strip_citations(text)
    lang = (data.get("lang") or "").strip().lower() or _detect_tts_lang(clean)

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        voice.synthesize(clean, tmp.name, lang=lang)
        return send_file(tmp.name, mimetype="audio/mpeg", as_attachment=False,
                         download_name="speech.mp3")
    except Exception as e:
        app.logger.exception("tts failed")
        return jsonify({"ok": False, "error": "tts_error", "detail": str(e)}), 500


# --------------------------------------------------------------- WhatsApp
@app.get("/whatsapp")
def wa_verify():
    """Meta calls this once to verify the webhook."""
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == WA_VERIFY_TOKEN:
        return challenge or "", 200
    return "forbidden", 403


def wa_send_text(to, body):
    if not (WA_TOKEN and WA_PHONE_ID):
        app.logger.warning("WhatsApp not configured; skipping send.")
        return
    url = f"{WA_API}/{WA_PHONE_ID}/messages"
    payload = {"messaging_product": "whatsapp", "to": to,
               "type": "text", "text": {"body": body[:4000]}}
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    try:
        requests.post(url, json=payload, headers=headers, timeout=20)
    except Exception as e:
        app.logger.error(f"WA send failed: {e}")


# Whether to reply to voice notes with a voice note back (else text only).
# Whether to reply to voice notes with a voice note back (else text only).
# Default is OFF (text-only). Set WA_VOICE_REPLY=1 in .env to enable spoken replies.
# Tolerant of quotes/whitespace/case so ".env" typos don't silently enable it.
_wa_voice_raw = os.environ.get("WA_VOICE_REPLY", "0").strip().strip('"').strip("'").lower()
WA_VOICE_REPLY = _wa_voice_raw in ("1", "true", "yes", "on")
print(f"[app] WA_VOICE_REPLY = {WA_VOICE_REPLY} (raw={_wa_voice_raw!r}) -> voice notes get "
      f"{'text + voice' if WA_VOICE_REPLY else 'text only'} replies")


def wa_download_media(media_id):
    """Download a WhatsApp media file (e.g. a voice note) to a temp file.
    Returns the local path, or None on failure."""
    if not WA_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        # Step 1: media_id -> a temporary download URL
        meta = requests.get(f"{WA_API}/{media_id}", headers=headers, timeout=20).json()
        url = meta.get("url")
        if not url:
            app.logger.error(f"WA media: no url for {media_id}: {meta}")
            return None
        # Step 2: download the actual bytes (same auth header required)
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)  # WA voice = OGG/Opus
        tmp.write(r.content)
        tmp.close()
        return tmp.name
    except Exception as e:
        app.logger.error(f"WA media download failed: {e}")
        return None


def wa_upload_audio(path):
    """Upload an audio file to WhatsApp; returns a media_id to send, or None."""
    if not (WA_TOKEN and WA_PHONE_ID):
        return None
    url = f"{WA_API}/{WA_PHONE_ID}/media"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        with open(path, "rb") as f:
            files = {"file": ("reply.mp3", f, "audio/mpeg")}
            data = {"messaging_product": "whatsapp", "type": "audio/mpeg"}
            r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        r.raise_for_status()
        return r.json().get("id")
    except Exception as e:
        app.logger.error(f"WA media upload failed: {e}")
        return None


def wa_send_audio(to, media_id):
    """Send a voice note (audio message) by media_id."""
    if not (WA_TOKEN and WA_PHONE_ID and media_id):
        return False
    url = f"{WA_API}/{WA_PHONE_ID}/messages"
    payload = {"messaging_product": "whatsapp", "to": to,
               "type": "audio", "audio": {"id": media_id}}
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        return r.ok
    except Exception as e:
        app.logger.error(f"WA send audio failed: {e}")
        return False


def _format_for_whatsapp(result):
    """Turn the structured result into a single WhatsApp-friendly text block."""
    if not result.get("ok"):
        return "Sorry, something went wrong. Please try again."
    answer = result.get("answer", "")
    if result.get("out_of_scope"):
        return answer
    pct = result.get("confidence_percent")
    foot = f"\n\nConfidence: {pct}% ({result.get('confidence_label')})"
    if result.get("consensus"):
        foot += " • All major madhhabs agree"
    foot += "\n_Always verify important matters with a qualified scholar._"
    return f"{answer}{foot}"


@app.post("/whatsapp")
def wa_inbound():
    """Inbound messages from Meta Cloud API."""
    data = request.get_json(silent=True) or {}
    try:
        entry = (data.get("entry") or [{}])[0]
        change = (entry.get("changes") or [{}])[0]
        value = change.get("value", {})
        messages = value.get("messages")
        if not messages:
            return "ok", 200  # delivery/status callback, ignore
        msg = messages[0]
        sender = msg.get("from")
        mtype = msg.get("type")

        if mtype == "text":
            question = msg["text"]["body"]
        elif mtype == "audio":
            # Voice note: download -> transcribe with Whisper -> answer.
            import voice
            media_id = (msg.get("audio") or {}).get("id")
            audio_path = wa_download_media(media_id) if media_id else None
            if not audio_path:
                wa_send_text(sender, "Sorry, I couldn't read that voice message. Please try again or send text.")
                return "ok", 200
            try:
                question, detected = voice.transcribe(audio_path)  # auto-detects en/ur/ar
            except Exception:
                app.logger.exception("whisper transcription failed")
                wa_send_text(sender, "Sorry, I couldn't understand that voice message. Please send your question as text.")
                return "ok", 200
            finally:
                try: os.remove(audio_path)
                except Exception: pass
            if not question.strip():
                wa_send_text(sender, "I couldn't hear a question in that voice note. Please try again.")
                return "ok", 200
            # Answer, reply with text, and (optionally) a spoken voice note back.
            result = rag_engine.answer_question(question, None)
            db.log_query(question, "General", result, channel="whatsapp",
                         extra={"voice": True, "language": detected})
            reply_text = _format_for_whatsapp(result)
            wa_send_text(sender, reply_text)
            if WA_VOICE_REPLY and result.get("ok"):
                try:
                    spoken = voice.strip_citations(result.get("answer", ""))
                    if spoken:
                        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False); tmp.close()
                        voice.synthesize(spoken, tmp.name, lang=detected)
                        mid = wa_upload_audio(tmp.name)
                        wa_send_audio(sender, mid)
                        try: os.remove(tmp.name)
                        except Exception: pass
                except Exception:
                    app.logger.exception("voice reply failed (text was still sent)")
            return "ok", 200
        else:
            wa_send_text(sender, "Please send your question as a text message.")
            return "ok", 200

        result = rag_engine.answer_question(question, None)  # madhhab selection via web UI; default General on WA
        db.log_query(question, "General", result, channel="whatsapp")
        wa_send_text(sender, _format_for_whatsapp(result))
    except Exception:
        app.logger.exception("whatsapp inbound failed")
    return "ok", 200


if __name__ == "__main__":
    start_warmup()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
