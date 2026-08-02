# -*- coding: utf-8 -*-
"""
db.py — optional MongoDB (Atlas) logging for the Islamic Fiqh chatbot.

Logs every question + answer + confidence + sources to a 'queries' collection.
Designed to FAIL SILENTLY: if Mongo is unreachable or MONGODB_URI is unset, the
chatbot keeps working normally — logging is a nice-to-have, never a blocker.
"""
import os
import datetime

_collection = None
_init_done = False


def _get_collection():
    """Lazily connect to Atlas. Returns the collection, or None if unavailable."""
    global _collection, _init_done
    if _init_done:
        return _collection
    _init_done = True

    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        print("[db] MONGODB_URI not set — query logging disabled.")
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")          # verify the connection works now
        db = client[os.environ.get("MONGODB_DB", "fiqh_chatbot")]
        _collection = db["queries"]
        print("[db] Connected to MongoDB Atlas — query logging ON.")
    except Exception as e:
        print(f"[db] MongoDB unavailable — logging disabled. ({e})")
        _collection = None
    return _collection


def log_query(question, madhhab, result, channel="web", extra=None):
    """
    Store one interaction. `result` is the dict from rag_engine.answer_question().
    `extra` is an optional dict of extra fields to merge in (e.g. voice/language).
    Never raises — any failure is swallowed so the response is never affected.
    """
    col = _get_collection()
    if col is None:
        return
    try:
        sources = result.get("sources", {}) or {}
        doc = {
            "timestamp":          datetime.datetime.utcnow(),
            "channel":            channel,                 # "web" / "whatsapp" / "voice_test"
            "question":           question,
            "madhhab":            madhhab or "General",
            "answer":             result.get("answer", ""),
            "mode":               result.get("mode"),      # fiqh / general / greeting / out_of_scope
            "out_of_scope":       result.get("out_of_scope", False),
            "confidence":         result.get("confidence"),
            "confidence_percent": result.get("confidence_percent"),
            "confidence_label":   result.get("confidence_label"),
            "consensus":          result.get("consensus", False),
            "topics":             [f.get("topic_id") for f in sources.get("fiqh", [])],
            "quran_refs":         [q.get("ref") for q in sources.get("quran", [])],
            "hadith_refs":        [h.get("ref") for h in sources.get("hadith", [])],
        }
        if extra:
            doc.update(extra)                              # e.g. {"voice": True, "language": "ur"}
        col.insert_one(doc)
    except Exception as e:
        print(f"[db] log_query failed (ignored): {e}")


def recent(limit=20):
    """Helper: fetch the most recent logged queries (for an analytics view)."""
    col = _get_collection()
    if col is None:
        return []
    try:
        return list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    except Exception:
        return []


def stats():
    """Helper: simple aggregate counts for a report/dashboard."""
    col = _get_collection()
    if col is None:
        return {}
    try:
        total = col.count_documents({})
        oos   = col.count_documents({"out_of_scope": True})
        by_channel = list(col.aggregate([{"$group": {"_id": "$channel", "n": {"$sum": 1}}}]))
        top_topics = list(col.aggregate([
            {"$unwind": "$topics"},
            {"$group": {"_id": "$topics", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}}, {"$limit": 10},
        ]))
        return {"total": total, "out_of_scope": oos,
                "by_channel": by_channel, "top_topics": top_topics}
    except Exception:
        return {}
