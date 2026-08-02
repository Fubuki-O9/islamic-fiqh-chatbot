# -*- coding: utf-8 -*-
"""
voice.py — server-side voice for WhatsApp (and a local test path).

  STT (speech -> text):  OpenAI Whisper (free, local, multilingual: en / ur / ar)
  TTS (text -> speech):  edge-tts        (free, Microsoft neural voices)

Both are imported lazily so the API server starts even if these heavy packages
aren't installed yet. Install when you're ready for voice:

    pip install openai-whisper edge-tts
    # Whisper also needs ffmpeg on the system:
    #   Windows:  winget install Gyan.FFmpeg     (or choco install ffmpeg)
    #   then restart the terminal so ffmpeg is on PATH
"""
import os
import asyncio

# ---- Whisper (STT) -------------------------------------------------------
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base")   # tiny|base|small|medium
# "base" is a good speed/quality balance. For better Urdu/Arabic use "small".
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper                      # lazy import
        print(f"[voice] loading Whisper '{WHISPER_SIZE}' (first time downloads the model)...")
        _whisper_model = whisper.load_model(WHISPER_SIZE)
        print("[voice] Whisper ready.")
    return _whisper_model


def transcribe(audio_path, language=None):
    """
    Speech -> text. `language=None` auto-detects (handles English, Urdu, Arabic).
    Returns (text, detected_language).
    """
    model = _load_whisper()
    result = model.transcribe(audio_path, language=language, fp16=False)
    return (result.get("text") or "").strip(), result.get("language")


# ---- edge-tts (TTS) ------------------------------------------------------
VOICE_EN = os.environ.get("TTS_VOICE_EN", "en-US-AriaNeural")
VOICE_AR = os.environ.get("TTS_VOICE_AR", "ar-SA-HamedNeural")
VOICE_UR = os.environ.get("TTS_VOICE_UR", "ur-PK-AsadNeural")


def _voice_for(lang):
    l = (lang or "").lower()
    if l.startswith("ar"):
        return VOICE_AR
    if l.startswith("ur"):
        return VOICE_UR
    return VOICE_EN


async def _synth(text, out_path, voice):
    import edge_tts                          # lazy import
    await edge_tts.Communicate(text, voice).save(out_path)


def synthesize(text, out_path, lang=None):
    """Text -> spoken audio (mp3) saved at out_path. Returns out_path."""
    voice = _voice_for(lang)
    asyncio.run(_synth(text, out_path, voice))
    return out_path


def strip_citations(text):
    """Remove [..] citation tags and markdown for cleaner spoken audio."""
    import re
    t = re.sub(r"\[[^\]]*\]", "", text or "")
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# ---- local self-test -----------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print("Transcribing:", path)
        text, lang = transcribe(path)
        print(f"  detected language: {lang}")
        print(f"  transcript: {text}")
        out = "tts_reply.mp3"
        synthesize("This is a test of the Islamic chatbot voice reply.", out, lang)
        print(f"  wrote TTS sample to {out}")
    else:
        print("Usage: python voice.py <audio-file>   (transcribes it, then writes a TTS sample)")
