# -*- coding: utf-8 -*-
"""
rag_engine.py — Islamic Fiqh RAG pipeline (refactored from the Colab notebook).

Loads the datasets once, builds BM25 + MiniLM hybrid indexes, and exposes a
single clean entry point:  answer_question(query, madhhab) -> dict

The dict is JSON-serialisable and shared by BOTH the web frontend (/ask) and
the WhatsApp webhook, so the two channels always behave identically.
"""
import os, re, time, json, string
import yaml
import numpy as np

# ---------------------------------------------------------------- config
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
FIQH_FILE   = os.environ.get("FIQH_FILE",   os.path.join(DATA_DIR, "fiqh_topics_complete_expanded.yaml"))
QURAN_FILE  = os.environ.get("QURAN_FILE",  os.path.join(DATA_DIR, "quran_trilingual_clean.json"))
HADITH_FILE = os.environ.get("HADITH_FILE", os.path.join(DATA_DIR, "sahih_bukhari_muslim.json"))
# Optional supplementary hadith from other authentic collections (Tirmidhi,
# Nasa'i, Abu Dawud) that aren't in the main Bukhari/Muslim corpus. Verified
# and added by the team from Sunnah.com. Loaded if present; ignored if absent.
OTHER_HADITH_FILE = os.environ.get("OTHER_HADITH_FILE", os.path.join(DATA_DIR, "other_hadith.json"))


def normalize_hadith_source(source):
    """Canonicalize hadith collection names so different spellings of the SAME
    collection match. The dataset and the topic references disagree on spelling
    ('Sahih Bukhari' vs 'Sahih al-Bukhari'), which silently broke lookups and
    dropped valid Bukhari/Muslim citations. Everything maps to one canonical
    form: 'Sahih Bukhari' or 'Sahih Muslim'. Non-Bukhari/Muslim collections are
    returned normalized but will simply not be found in the dataset (intended:
    only Bukhari & Muslim are cited)."""
    s = re.sub(r"\s+", " ", (source or "").strip().lower())
    s = s.replace("-", " ").replace("'", "").replace("’", "")
    # strip common prefixes/particles
    s = s.replace("sahih al ", "sahih ").replace("al bukhari", "bukhari")
    s = s.replace("sahih al-bukhari", "sahih bukhari")
    if "bukhar" in s:
        return "Sahih Bukhari"
    if "muslim" in s:
        return "Sahih Muslim"
    # leave others as title-cased original (won't match the BM-only dataset)
    return (source or "").strip()

EXAMPLE_FILE= os.environ.get("EXAMPLE_FILE",os.path.join(DATA_DIR, "topic_examples_expanded.yaml"))

# Fine-tuned model folder (from finetune_islamic_minilm.ipynb). Falls back to
# the base MiniLM if the folder is not present — so the server always runs.
FINETUNED_DIR = os.environ.get("MINILM_DIR", os.path.join(HERE, "models", "islamic_minilm"))
# Fine-tuned model published on the Hugging Face Hub. If the local FINETUNED_DIR
# is absent, sentence-transformers downloads this automatically on first run.
# Defaults to the published model; override with your own Hub repo id if you fork.
HF_MODEL      = os.environ.get("HF_MODEL", "FubukiO9/fiqh-minilm").strip()
# Multilingual base so Urdu/Roman-Urdu/Arabic queries embed meaningfully even if
# the fine-tuned folder is absent. (all-MiniLM-L6-v2 is English-only and mangles
# non-English input, which caused wrong retrieval + wrong-language answers.)
BASE_MODEL    = os.environ.get("BASE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

ALPHA       = float(os.environ.get("HYBRID_ALPHA", "0.45"))   # BM25 weight; (1-alpha) = semantic
# Minimum similarity for a NON-curated (semantic-search) citation to be shown.
# Curated references from the dataset always pass; only guessed ones are gated.
HADITH_MIN_SCORE = float(os.environ.get("HADITH_MIN_SCORE", "0.50"))
QURAN_MIN_SCORE  = float(os.environ.get("QURAN_MIN_SCORE", "0.45"))
# Hadith citations are only trustworthy when they come from curated, verified
# references. A semantic search over ~15k hadiths can return a same-book-but-WRONG
# hadith (verified case: Bukhari #137 is about passing wind, not bleeding), which
# is unacceptable for a religious citation. So the search fallback is OFF by default.
ENABLE_HADITH_SEARCH = os.environ.get("ENABLE_HADITH_SEARCH", "false").lower() == "true"
# General-knowledge mode grounds its citations in the datasets too (real verses /
# hadith only). These gate how relevant a retrieved reference must be to be offered.
GENERAL_QURAN_MIN  = float(os.environ.get("GENERAL_QURAN_MIN", "0.35"))
GENERAL_HADITH_MIN = float(os.environ.get("GENERAL_HADITH_MIN", "0.45"))
# Below this topic-classifier score, treat the question as general (not a curated
# fiqh edge-case). Raised to 0.65 so weak/ambiguous matches (e.g. Roman-Urdu
# phrasings the classifier isn't sure about) fall to the general path and answer
# correctly, instead of grasping at the wrong curated topic with false confidence.
# Lower it if genuine fiqh questions start slipping into general mode.
GENERAL_IF_BELOW   = float(os.environ.get("GENERAL_IF_BELOW", "0.65"))
# In general-knowledge answers, a matched topic's VERIFIED references become
# citation candidates once its score clears this gate. Lower = more general
# answers can surface a real citation (the uncited-tag stripper still prevents
# any fabricated reference, so lowering this never invents a source).
GENERAL_CITE_MIN   = float(os.environ.get("GENERAL_CITE_MIN", "0.35"))
# Below this confidence %, decline and refer to a trusted scholarly resource
# rather than generating a possibly-wrong ruling (configurable via env).
LOW_CONFIDENCE_PCT = int(os.environ.get("LOW_CONFIDENCE_PCT", "50"))
ISLAMQA_URL        = os.environ.get("ISLAMQA_URL", "https://islamqa.info/en")
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
]

# Lazy-initialised globals (filled by init())
_READY = False
fiqh_topics = quran_data = hadith_data = topic_examples = None
quran_lookup = hadith_lookup = None
bm25_fiqh = bm25_hadith = bm25_quran = None
fiqh_texts = hadith_texts = quran_texts = None
embedder = None
fiqh_embeddings = example_embeddings = None
example_topic_ids = None
stop_words = set()
_genai_client = None


# ---------------------------------------------------------------- text utils
def preprocess(text):
    """Lowercase, strip punctuation (keep Arabic), tokenise, drop stopwords."""
    from nltk.tokenize import word_tokenize
    text = (text or "").lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def fiqh_topic_to_text(topic):
    parts = [
        topic.get("topic_name", ""),
        topic.get("description", ""),
        topic.get("topic_id", "").replace("_", " "),
    ]
    for _m, rd in (topic.get("rulings", {}) or {}).items():
        if isinstance(rd, dict):
            parts.append(rd.get("ruling", ""))
            parts.append(rd.get("explanation", ""))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------- init
def init(verbose=True):
    """Load data + build indexes. Safe to call multiple times (idempotent)."""
    global _READY, fiqh_topics, quran_data, hadith_data, topic_examples
    global quran_lookup, hadith_lookup, bm25_fiqh, bm25_hadith, bm25_quran
    global fiqh_texts, hadith_texts, quran_texts, embedder
    global fiqh_embeddings, example_embeddings, example_topic_ids, stop_words
    if _READY:
        return

    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    if verbose: print("Loading datasets...")
    with open(FIQH_FILE, encoding="utf-8") as f:
        fiqh_topics = yaml.safe_load(f)
    with open(QURAN_FILE, encoding="utf-8") as f:
        quran_data = json.load(f)
    with open(HADITH_FILE, encoding="utf-8") as f:
        hadith_data = json.load(f)
    with open(EXAMPLE_FILE, encoding="utf-8") as f:
        topic_examples = yaml.safe_load(f)

    quran_lookup  = {(v["chapter"], v["ayah"]): v for v in quran_data}
    hadith_lookup = {(normalize_hadith_source(h["source"]), str(h["hadith_no"])): h for h in hadith_data}

    # Merge in supplementary hadith from other collections, if the file exists.
    # These become citable exactly like Bukhari/Muslim entries (same lookup key).
    try:
        if os.path.exists(OTHER_HADITH_FILE):
            with open(OTHER_HADITH_FILE, encoding="utf-8") as f:
                other = json.load(f)
            added = 0
            for h in other:
                src = (h.get("source") or "").strip()
                no  = str(h.get("hadith_no", "")).strip()
                if not src or not no:
                    continue
                # keep original source name (Tirmidhi/Nasa'i/Abu Dawud) — do NOT
                # normalize these to Bukhari/Muslim; just index by exact source.
                hadith_lookup[(src, no)] = {
                    "source": src, "hadith_no": h.get("hadith_no"),
                    "chapter": h.get("chapter", ""),
                    "text_en": h.get("text_en", ""), "text_ar": h.get("text_ar", ""),
                }
                added += 1
            if added:
                print(f"[rag] Loaded {added} supplementary hadith from other_hadith.json")
    except Exception as e:
        print(f"[rag] other_hadith.json not loaded ({e}) — continuing without it.")

    from rank_bm25 import BM25Okapi
    if verbose: print("Building BM25 indexes...")
    fiqh_texts   = [fiqh_topic_to_text(t) for t in fiqh_topics]
    bm25_fiqh    = BM25Okapi([preprocess(t) for t in fiqh_texts])
    hadith_texts = [h.get("text_en", "") for h in hadith_data]
    bm25_hadith  = BM25Okapi([preprocess(t) for t in hadith_texts])
    quran_texts  = [v.get("english", "") for v in quran_data]
    bm25_quran   = BM25Okapi([preprocess(t) for t in quran_texts])

    from sentence_transformers import SentenceTransformer
    # Model resolution order:
    #   1) local fine-tuned folder (fastest, offline) if present
    #   2) fine-tuned model on the Hugging Face Hub (HF_MODEL) — auto-downloads
    #   3) multilingual base model (always available)
    if os.path.isdir(FINETUNED_DIR):
        if verbose: print(f"Loading fine-tuned MiniLM from {FINETUNED_DIR}")
        embedder = SentenceTransformer(FINETUNED_DIR)
    elif HF_MODEL:
        if verbose: print(f"Loading fine-tuned MiniLM from Hugging Face: {HF_MODEL}")
        try:
            embedder = SentenceTransformer(HF_MODEL)
        except Exception as e:
            if verbose: print(f"Could not load {HF_MODEL} ({e}); using base {BASE_MODEL}")
            embedder = SentenceTransformer(BASE_MODEL)
    else:
        if verbose: print(f"Fine-tuned model not found — using base {BASE_MODEL}")
        embedder = SentenceTransformer(BASE_MODEL)

    if verbose: print("Embedding fiqh topics + topic examples...")
    fiqh_embeddings = embedder.encode(fiqh_texts, convert_to_tensor=True, show_progress_bar=False)
    ex_texts, example_topic_ids = [], []
    for tid, exs in topic_examples.items():
        for ex in exs:
            ex_texts.append(ex); example_topic_ids.append(tid)
    example_embeddings = embedder.encode(ex_texts, convert_to_tensor=True, show_progress_bar=False)

    _READY = True
    if verbose:
        print(f"Engine ready: {len(fiqh_topics)} topics, {len(quran_data):,} ayahs, "
              f"{len(hadith_data):,} hadiths, {len(ex_texts):,} examples.")


# ---------------------------------------------------------------- retrieval
def classify_topic(query, top_n=3):
    from sentence_transformers import util
    q = embedder.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(q, example_embeddings)[0].cpu().numpy()
    scores = {}
    for i, tid in enumerate(example_topic_ids):
        if tid not in scores or sims[i] > scores[tid]:
            scores[tid] = float(sims[i])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def retrieve_fiqh_ruling(query, madhhab=None, top_k=3, alpha=ALPHA):
    from sentence_transformers import util
    tokens = preprocess(query)
    bm = bm25_fiqh.get_scores(tokens)
    bm_max = bm.max()
    bm_norm = bm / bm_max if bm_max > 0 else bm
    q = embedder.encode(query, convert_to_tensor=True)
    sem = util.cos_sim(q, fiqh_embeddings)[0].cpu().numpy()
    hybrid = alpha * bm_norm + (1 - alpha) * sem
    ranked = sorted(enumerate(hybrid), key=lambda x: x[1], reverse=True)[:top_k]

    out = []
    for idx, score in ranked:
        topic = fiqh_topics[idx]
        rulings = topic.get("rulings", {}) or {}
        mk = madhhab.lower() if madhhab else None
        if mk == "shafi'i":
            mk = "shafi"
        ruling = {mk: rulings[mk]} if (mk and mk in rulings) else rulings
        out.append({
            "type": "fiqh",
            "topic_id": topic["topic_id"],
            "topic_name": topic["topic_name"],
            "description": topic.get("description", ""),
            "ruling": ruling,
            "consensus": topic.get("consensus", False),
            "references": topic.get("references", {}) or {},
            "scholar_review": topic.get("scholar_review"),  # surfaced for transparency
            "score": float(score),
        })
    return out


def _fiqh_entry_by_id(topic_id, madhhab=None, score=0.0):
    """Build a fiqh-result dict (same shape as retrieve_fiqh_ruling) for one topic."""
    for t in fiqh_topics:
        if t["topic_id"] == topic_id:
            rulings = t.get("rulings", {}) or {}
            mk = madhhab.lower() if madhhab else None
            if mk == "shafi'i":
                mk = "shafi"
            ruling = {mk: rulings[mk]} if (mk and mk in rulings) else rulings
            return {
                "type": "fiqh", "topic_id": t["topic_id"], "topic_name": t["topic_name"],
                "description": t.get("description", ""), "ruling": ruling,
                "consensus": t.get("consensus", False),
                "references": t.get("references", {}) or {},
                "scholar_review": t.get("scholar_review"), "score": float(score),
            }
    return None


def retrieve_hadith(query, top_k=3, bm25_candidates=50, alpha=ALPHA):
    from sentence_transformers import util
    tokens = preprocess(query)
    bm = bm25_hadith.get_scores(tokens)
    top = np.argsort(bm)[::-1][:bm25_candidates]
    bm_max = bm[top[0]] if len(top) else 1
    cand = [hadith_texts[i] for i in top]
    cemb = embedder.encode(cand, convert_to_tensor=True)
    q = embedder.encode(query, convert_to_tensor=True)
    sem = util.cos_sim(q, cemb)[0].cpu().numpy()
    bm_norm = np.array([bm[i] / bm_max if bm_max else 0 for i in top])
    hybrid = alpha * bm_norm + (1 - alpha) * sem
    local = np.argsort(hybrid)[::-1][:top_k]
    out = []
    for li in local:
        h = hadith_data[top[li]]
        out.append({
            "type": "hadith", "source": h["source"], "hadith_no": h["hadith_no"],
            "chapter": h.get("chapter", ""), "text_en": h.get("text_en", ""),
            "text_ar": h.get("text_ar", ""), "score": float(hybrid[li]),
        })
    return out


def retrieve_quran(query, fiqh_refs=None, top_k=3, bm25_candidates=30, alpha=ALPHA):
    from sentence_transformers import util
    direct = []
    if fiqh_refs:
        for ref in fiqh_refs:
            ch, ay = ref.get("chapter"), ref.get("ayah")
            v = quran_lookup.get((ch, ay)) if (ch and ay) else None
            if v:
                direct.append({
                    "type": "quran", "chapter": ch, "ayah": ay,
                    "surah_name_en": v.get("surah_name_en", ""),
                    "arabic": v.get("arabic", ""), "english": v.get("english", ""),
                    "urdu": v.get("urdu", ""), "score": 1.0,
                })
    if direct:
        return direct[:top_k]

    tokens = preprocess(query)
    bm = bm25_quran.get_scores(tokens)
    top = np.argsort(bm)[::-1][:bm25_candidates]
    bm_max = bm[top[0]] if len(top) else 1
    cand = [quran_texts[i] for i in top]
    cemb = embedder.encode(cand, convert_to_tensor=True)
    q = embedder.encode(query, convert_to_tensor=True)
    sem = util.cos_sim(q, cemb)[0].cpu().numpy()
    bm_norm = np.array([bm[i] / bm_max if bm_max else 0 for i in top])
    hybrid = alpha * bm_norm + (1 - alpha) * sem
    local = np.argsort(hybrid)[::-1][:top_k]
    out = []
    for li in local:
        v = quran_data[top[li]]
        out.append({
            "type": "quran", "chapter": v["chapter"], "ayah": v["ayah"],
            "surah_name_en": v.get("surah_name_en", ""), "arabic": v.get("arabic", ""),
            "english": v.get("english", ""), "urdu": v.get("urdu", ""),
            "score": float(hybrid[li]),
        })
    return out


def compute_confidence(fiqh, hadith, quran, topic_score=0.0, topic_gap=0.0):
    """
    Phrasing-stable, banded confidence. It scores the QUALITIES of the answer —
    how clearly it matches a curated topic, whether the schools agree, and whether
    it has scripture backing — rather than the raw embedding similarity, which
    wobbles a few points with wording. So two phrasings of the same consensus
    question land on the same number.
    """
    # Topic-match strength as discrete bands (not the raw score) -> stable base.
    if   topic_score >= 0.70: score = 0.62     # strong, unambiguous match
    elif topic_score >= 0.55: score = 0.50     # good match
    else:                      score = 0.42     # borderline (rare on the fiqh path)

    # Consensus is a stable yes/no — the biggest single lever.
    if fiqh and fiqh[0].get("consensus"):
        score += 0.20

    # Scripture backing by PRESENCE (a topic's curated refs don't change with wording).
    if quran:  score += 0.08
    if hadith: score += 0.08

    if score != score:   # NaN guard
        score = 0.0
    return round(min(score, 0.99), 3)


def confidence_to_percent(score):
    # Map 0–1 to 25–99%. Never 100 (Islamic humility — always defer to scholars);
    # floor low enough that weak/uncertain answers read as genuinely Low.
    pct = int(min(25 + (score * 74), 99))
    if   pct >= 85: label = "Very High"
    elif pct >= 70: label = "High"
    elif pct >= 50: label = "Medium"
    else:           label = "Low"
    return pct, label


# ---------------------------------------------------------------- generation
def _client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        if not GEMINI_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _genai_client = genai.Client(api_key=GEMINI_KEY)
    return _genai_client


def gemini_generate(prompt):
    client = _client()
    last = None
    for model in GEMINI_MODELS:
        try:
            r = client.models.generate_content(model=model, contents=prompt)
            return r.text.strip()
        except Exception as e:
            err = str(e); last = e
            if "503" in err or "UNAVAILABLE" in err:
                time.sleep(2); continue
            if "429" in err or "QUOTA" in err.upper():
                continue
            raise
    raise RuntimeError(f"All Gemini models unavailable: {last}")


def build_rag_prompt(query, madhhab, fiqh, hadith, quran, target_language=None):
    parts = []
    if fiqh:
        parts.append("=== FIQH RULINGS FROM KNOWLEDGE BASE ===")
        for r in fiqh:
            parts.append(f'\n[{r["topic_name"]}] Topic ID: {r["topic_id"]}')
            parts.append(f'Description: {r["description"]}')
            parts.append(f'Scholarly Consensus: {"Yes" if r["consensus"] else "No - scholars differ"}')
            for mn, rd in r["ruling"].items():
                label = "Shafi'i" if mn == "shafi" else mn.capitalize()
                parts.append(f'  [{label}] Ruling: {rd.get("ruling","")} | Certainty: {rd.get("certainty","")}')
                parts.append(f'  Explanation: {rd.get("explanation","")}')
    if quran:
        parts.append("\n=== QURAN REFERENCES ===")
        for v in quran:
            parts.append(f'\n[Quran {v["chapter"]}:{v["ayah"]}] Surah {v["surah_name_en"]}')
            parts.append(f'Arabic: {v["arabic"]}')
            parts.append(f'English: {v["english"]}')
    if hadith:
        with_text = [h for h in hadith if h.get("text_en")]
        if with_text:
            parts.append("\n=== HADITH REFERENCES ===")
            for h in with_text:
                t = h["text_en"][:600] + "..." if len(h["text_en"]) > 600 else h["text_en"]
                parts.append(f'\n[{h["source"]} #{h["hadith_no"]}] Chapter: {h["chapter"]}')
                parts.append(t)
    context = "\n".join(parts)

    madhhab_instruction = ""
    if madhhab and madhhab != "General":
        madhhab_instruction = (f"The user follows the **{madhhab}** school of thought. Focus your answer "
                               f"on the {madhhab} ruling but mention other schools if they differ significantly.")

    lang_directive = ""
    lang_reminder = ""
    if target_language:
        lang_directive = (f"### OUTPUT LANGUAGE (HIGHEST PRIORITY) ###\n"
                          f"The user wrote their question in {target_language}. "
                          f"You MUST write your ENTIRE reply in {target_language}, and in NO other language. "
                          f"This overrides everything else. The reference material below may be in a different "
                          f"language \u2014 ignore its language and still answer in {target_language}. "
                          f"Only bracketed citations like [Quran 4:101] or [Sahih Bukhari #1] stay in Latin letters. "
                          f"Never reply in Hindi or Devanagari. Never say you cannot write a language \u2014 just write it.\n")
        lang_reminder = (f"\n\nFINAL REMINDER: Your entire answer above must be written in {target_language} only "
                         f"(citations in brackets excepted). If any part is in another language, rewrite it in {target_language}.")

    return f"""{lang_directive}You are an Islamic knowledge assistant providing accurate, respectful Islamic guidance.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using the information in the KNOWLEDGE BASE CONTEXT below.
2. Do NOT add any information from outside the provided context.
3. Citations:
   - Cite the FIQH TOPIC name ONCE, the first time you introduce the ruling, e.g. [Combining Prayers During Travel]. Do NOT repeat the same topic tag again in later sentences.
   - Cite Quran verses every time you use them: [Quran 4:101]
   - Cite Hadiths every time you use them: [Sahih Muslim #705] or [Sahih Bukhari #1]
   - Never place the same topic tag in two consecutive sentences.
4. After the first topic citation, support further statements with Quran/Hadith tags rather than repeating the topic name.
5. If the context does not answer the question, respond exactly: "This requires consultation with a qualified scholar." (translated into {target_language or 'the user language'}).
6. Do NOT issue personal fatwas or rulings beyond what the sources state.
7. If scholars disagree (ikhtilaf), clearly state the different positions with their citations.
8. Maintain a respectful, neutral, scholarly tone.
{madhhab_instruction}

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END OF CONTEXT ---

User Question: {query}
Remember: cite the fiqh topic only once (at first mention), then use Quran/Hadith citations. No repeated topic tags.{lang_reminder}
Answer:"""


# ---------------------------------------------------------------- public API
def _collapse_topic_tags(text):
    """
    Keep each fiqh-TOPIC tag only the first time it appears; drop later repeats.
    Quran tags ([Quran ...]) and Hadith tags (anything containing '#') are NEVER
    touched — those are real sources and are allowed to repeat.
    """
    if not text:
        return text
    seen = set()
    def repl(m):
        inner = m.group(1).strip()
        if inner.lower().startswith("quran") or "#" in inner:
            return m.group(0)          # scripture citation — always keep
        key = inner.lower()
        if key in seen:
            return ""                  # duplicate topic tag — remove
        seen.add(key)
        return m.group(0)
    out = re.sub(r"\[([^\]]+)\]", repl, text)
    out = re.sub(r"[ \t]{2,}", " ", out)        # tidy double spaces
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)   # tidy space before punctuation
    return out.strip()


def _is_refusal(text):
    """True when the model returned only the scholar-referral line (couldn't answer)."""
    if not text:
        return True
    t = text.strip().lower()
    return "requires consultation with a qualified scholar" in t and len(text) < 300


def general_generate(query, madhhab=None):
    """Answer a general (non-curated-fiqh) Islamic question from well-established knowledge."""
    m = f" The user follows the {madhhab} school of thought." if (madhhab and madhhab != "General") else ""
    prompt = f"""You are a knowledgeable, careful Islamic assistant answering a general question.
Answer using well-established, mainstream Islamic knowledge, clearly and concisely.{m}

Guidelines:
- For widely-agreed basics that every school accepts (e.g. the five daily prayers, the pillars of Islam, that fasting is in Ramadan), answer directly and confidently — these are not matters of dispute.
- If the matter is a genuine point of scholarly disagreement or needs a personalised ruling, give the general view and add that a qualified scholar should be consulted for specifics.
- Do NOT fabricate Quran or Hadith reference numbers. You may mention a reference only if you are certain it is correct; otherwise describe it without inventing a number.
- If the question is NOT about Islam, politely reply that you can only help with Islamic questions.

Question: {query}
Answer:"""
    return gemini_generate(prompt)


def _strip_uncited_tags(answer, allowed):
    """Remove any [Quran ..]/[Source #..] tag the model wrote that we did NOT provide
    (guards against the model inventing a citation that isn't in our datasets)."""
    if not answer:
        return answer
    def repl(m):
        tag = m.group(0)
        # keep only real citation-shaped tags that are in the allowed set
        is_cite = tag.lower().startswith("[quran") or "#" in tag
        if not is_cite:
            return tag                      # not a citation (leave alone)
        return tag if tag in allowed else ""
    out = re.sub(r"\[[^\]]+\]", repl, answer)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


def build_general_prompt(query, madhhab, quran, hadith, target_language=None):
    parts = []
    if quran:
        parts.append("=== QURAN VERSES (you may cite these) ===")
        for v in quran:
            parts.append(f'[Quran {v["chapter"]}:{v["ayah"]}] Surah {v["surah_name_en"]}: {v["english"]}')
    if hadith:
        parts.append("\n=== HADITH (you may cite these) ===")
        for h in hadith:
            t = h["text_en"][:400] + ("..." if len(h["text_en"]) > 400 else "")
            parts.append(f'[{h["source"]} #{h["hadith_no"]}]: {t}')
    context = "\n".join(parts) if parts else "(no specific references were retrieved for this question)"
    m = f" The user follows the {madhhab} school of thought." if (madhhab and madhhab != "General") else ""
    lang_directive = ""
    lang_reminder = ""
    if target_language:
        lang_directive = (f"### OUTPUT LANGUAGE (HIGHEST PRIORITY) ###\n"
                          f"The user wrote in {target_language}. Write your ENTIRE reply in {target_language} and no other "
                          f"language. This overrides everything else. The references below may be in another language \u2014 "
                          f"ignore that and still answer in {target_language}. Only bracketed citations stay in Latin letters. "
                          f"Never reply in Hindi/Devanagari. Never claim you cannot write a language \u2014 just write it.\n\n")
        lang_reminder = (f"\nFINAL REMINDER: the entire answer must be in {target_language} only (bracketed citations excepted).")
    return f"""{lang_directive}You are a knowledgeable, careful Islamic assistant answering a general question.{m}
Give a clear, complete and helpful answer (a few sentences; use a short list if it fits).

CITATION RULES (critical):
- You MAY cite ONLY the references listed in the CONTEXT below, using their exact tags, e.g. [Quran 4:103] or [Sahih Bukhari #8].
- Where a listed reference supports part of your answer (for example, a verse commanding the establishment of prayer), DO cite it inline.
- If a listed reference is not relevant, simply don't cite it. NEVER invent or cite anything not in the CONTEXT.
- If the CONTEXT lists NO references, do NOT cite any Quran verse or Hadith number at all. Speak in general terms instead (e.g. "the Sunnah indicates...", "scholars hold...") WITHOUT any bracketed citation or specific number. Never write a Quran or Hadith number from memory.
- It is fine if the exact detail (e.g. an exact number) is established by the Sunnah rather than a verse — still answer it confidently and cite whatever listed reference is genuinely relevant.
- For widely-agreed basics (the five daily prayers, the pillars of Islam, etc.), answer directly and confidently — these are not disputed.
- When the question asks "how many" of something, NAME each item, not just the count (e.g. list the five daily prayers: Fajr, Dhuhr, Asr, Maghrib, Isha; or name the pillars).
- Treat questions about whether something is permissible (halal/haram), worship, transactions, money, ethics, food, family, or Muslim daily life as Islamic questions and ANSWER them — applying general Islamic principles (e.g. avoiding gambling/maysir, interest/riba, or harm) where relevant.
- ONLY if the question is clearly unrelated to Islam or Muslim life (e.g. secular trivia, sports scores, product reviews) should you reply that you can only help with Islamic questions.

--- CONTEXT (the only references you may cite) ---
{context}
--- END CONTEXT ---

Question: {query}
Answer:{lang_reminder}"""


def _curated_candidates(topic_id):
    """Verified Quran/Hadith references from a curated topic, as citation candidates."""
    entry = _fiqh_entry_by_id(topic_id)
    if not entry:
        return [], []
    refs = entry.get("references", {}) or {}
    qcands = []
    for r in (refs.get("quran") or []):
        v = quran_lookup.get((r.get("chapter"), r.get("ayah")))
        if v:
            qcands.append({"type": "quran", "chapter": v["chapter"], "ayah": v["ayah"],
                           "surah_name_en": v.get("surah_name_en", ""), "arabic": v.get("arabic", ""),
                           "english": v.get("english", ""), "urdu": v.get("urdu", ""), "score": 1.0})
    hcands = []
    for r in (refs.get("hadith") or []):
        src, no = normalize_hadith_source(r.get("source", "")), r.get("hadith_no", "")
        h = hadith_lookup.get((src, str(no)))
        if h:
            hcands.append({"type": "hadith", "source": h["source"], "hadith_no": h["hadith_no"],
                           "chapter": h.get("chapter", ""), "text_en": h.get("text_en", ""),
                           "text_ar": h.get("text_ar", ""), "score": 1.0})
        # Non-Bukhari/Muslim references (not in the corpus) are intentionally
        # skipped — we only cite hadith we can show from Sahih Bukhari/Muslim.
    return qcands, hcands


def _is_ruling_request(query):
    """True when the question asks for a permissibility/fiqh RULING (halal/haram,
    'is it allowed', 'can men/women...', 'is it permissible') rather than a basic
    fact. Used to decide that, without any verified reference, we must refer the
    user to a scholar/IslamQA instead of extrapolating a novel ruling."""
    q = " " + re.sub(r"[^\w\s]", " ", query.lower()).strip() + " "
    ruling_markers = [
        "permissible", "permissable", "impermissible", "allowed", "allow ",
        "halal", "haram", "haraam", "forbidden", "prohibited", "sinful",
        "is it ok", "is it okay", "is it fine", "is it wrong", "is it a sin",
        "can i ", "can we ", "can men ", "can women ", "can a man ", "can a woman ",
        "can muslims", "am i allowed", "are we allowed", "may i ", "should i wear",
        "jaiz", "jayez", "gunah", "napak",
    ]
    return any(m in q for m in ruling_markers)


def _is_sensitive_question(query):
    """True for grave, contested, or highly personal matters that a small
    automated system must NOT rule on \u2014 e.g. violence, sexuality, apostasy,
    takfir, or explicit acts. These are ALWAYS referred to a qualified scholar
    regardless of any weak topic match, because a wrong or oversimplified answer
    here can cause real harm. This is a safety backstop, not a moral judgement."""
    q = " " + re.sub(r"[^\w\s]", " ", (query or "").lower()).strip() + " "
    sensitive = [
        # violence / harm to self or others (English)
        " murder", " kill ", " killing", " suicide", " terroris", " bomb", " jihad ",
        " behead", " stone to death", " honor kill", " honour kill", " violence",
        # sexuality / relationships (English)
        " gay", " lesbian", " homosexual", " lgbt", " transgender", " trans ",
        " zina", " adultery", " fornicat", " sex ", " sexual", " masturbat",
        # creed-critical / interpersonal grave matters (English)
        " apostasy", " apostate", " leave islam", " kafir", " takfir", " kill an apostate",
        " divorce my", " triple talaq", " abortion", " euthanasia",
        # Roman Urdu markers
        " qatal", " khudkushi", " khudkhushi", " samlaingik", " hum jinsi",
        " zina", " talaq", " isqat", " isqaat e haml", " murtad",
        # Urdu / Arabic script markers
        "\u0642\u062a\u0644",            # qatl (murder/killing)
        "\u062e\u0648\u062f\u06a9\u0634\u06cc",  # khudkushi (suicide)
        "\u0632\u0646\u0627",            # zina
        "\u0637\u0644\u0627\u0642",      # talaq (divorce)
        "\u0627\u0633\u0642\u0627\u0637",  # isqat (abortion)
        "\u0645\u0631\u062a\u062f",      # murtad (apostate)
        "\u0627\u0646\u062a\u062d\u0627\u0631",  # intihar (suicide, Arabic)
        "\u0634\u0630\u0648\u0630",      # shudhudh (homosexuality, Arabic)
    ]
    return any(m in q for m in sensitive)


def _general_result(query, madhhab=None, topic_id=None, topic_score=0.0):
    """General-knowledge answer.

    CITATION INTEGRITY: on this path we do NOT let the model cite blind
    keyword-search hits — a hadith that merely shares words with the question
    (e.g. an oaths hadith surfacing for a prayer question) is not a valid proof
    and produced wrong citations like "Sahih Muslim #1671". So the ONLY citable
    references here are the *curated, verified* refs attached to a matched topic
    in the dataset. If no topic is close enough, the answer cites nothing and
    speaks in general terms instead of showing an unverifiable number.
    """
    quran_cands, hadith_cands = [], []

    # Only offer a topic's VERIFIED references (never raw search candidates).
    if topic_id and topic_score >= GENERAL_CITE_MIN:
        quran_cands, hadith_cands = _curated_candidates(topic_id)

    prompt = build_general_prompt(query, madhhab, quran_cands, hadith_cands,
                                  target_language=_detect_language(query))
    answer = gemini_generate(prompt)

    # Only the references we actually provided are valid citations; strip the rest.
    allowed = set(f'[Quran {v["chapter"]}:{v["ayah"]}]' for v in quran_cands)
    allowed |= set(f'[{h["source"]} #{h["hadith_no"]}]' for h in hadith_cands)
    answer = _strip_uncited_tags(answer, allowed)
    low = answer.lower()

    non_islamic = ("only help with islamic" in low or "only answer islamic" in low) and len(answer) < 240

    # Show ONLY references that survived as inline citations.
    shown_quran  = [v for v in quran_cands if f'[quran {v["chapter"]}:{v["ayah"]}]' in low]
    shown_hadith = [h for h in hadith_cands if f'[{h["source"]} #{h["hadith_no"]}]'.lower() in low]

    # SAFETY: a permissibility/ruling question that ended up with NO verified
    # reference means we'd only be extrapolating a novel ruling (e.g. "is a gold
    # car haram?"). That is exactly what this project must not do — refer the
    # user to a qualified scholar / IslamQA instead of inventing a ruling.
    if (not non_islamic and _is_ruling_request(query)
            and not shown_quran and not shown_hadith):
        pct, label = confidence_to_percent(0.30)
        return _low_confidence_result(query, madhhab, pct, label)

    confidence = 0.30 if non_islamic else 0.62
    pct, label = confidence_to_percent(confidence)
    return {
        "ok": True, "mode": "general", "general_knowledge": True,
        "out_of_scope": non_islamic,
        "answer": answer,
        "confidence": confidence, "confidence_percent": pct, "confidence_label": label,
        "consensus": False,
        "sources": {
            "fiqh": [],
            "quran":  [{"ref": f'{v["chapter"]}:{v["ayah"]}', "surah": v.get("surah_name_en", ""),
                        "arabic": v.get("arabic", ""), "english": v.get("english", ""),
                        "urdu": v.get("urdu", "")} for v in shown_quran],
            "hadith": [{"ref": f'{h["source"]} #{h["hadith_no"]}', "chapter": h.get("chapter", ""),
                        "text_ar": h.get("text_ar", ""), "text_en": h.get("text_en", "")} for h in shown_hadith],
        },
    }


def _lang_key(query):
    """Short language key for selecting fixed (non-LLM) messages: 'ur', 'ar',
    'roman', or 'en'. Mirrors _detect_language but returns a compact code."""
    tl = _detect_language(query)
    if "Urdu script" in tl: return "ur"
    if tl == "Arabic":      return "ar"
    if "Roman Urdu" in tl:  return "roman"
    return "en"


# Fixed messages, localized so a referral/greeting/rephrase reply always matches
# the user's language. Citations/URLs stay as-is (Latin) inside these.
_MSG = {
    "referral": {
        "en": ("I couldn't find a reliable, well-sourced answer to this question in my "
               "knowledge base, so I won't guess \u2014 and some matters genuinely depend on "
               "the person's situation and require a scholar's judgement.\n\n"
               "Please consult a qualified scholar, or look this up on a trusted resource such as {url}"),
        "ur": ("\u0645\u062c\u06be\u06d2 \u0627\u067e\u0646\u06d2 \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0645\u06cc\u06ba \u0627\u0633 \u0633\u0648\u0627\u0644 \u06a9\u0627 \u0645\u0633\u062a\u0646\u062f \u0627\u0648\u0631 \u0642\u0627\u0628\u0644\u0650 \u0627\u0639\u062a\u0645\u0627\u062f \u062c\u0648\u0627\u0628 \u0646\u06c1\u06cc\u06ba \u0645\u0644\u0627\u060c \u0627\u0633 \u0644\u06cc\u06d2 \u0645\u06cc\u06ba \u0627\u0646\u062f\u0627\u0632\u06c1 \u0646\u06c1\u06cc\u06ba \u0644\u06af\u0627\u0624\u06ba \u06af\u0627\u06d4 \u06a9\u0686\u06be \u0645\u0633\u0627\u0626\u0644 \u0627\u0646\u0633\u0627\u0646 \u06a9\u06d2 \u062d\u0627\u0644\u0627\u062a \u067e\u0631 \u0645\u0646\u062d\u0635\u0631 \u06c1\u0648\u062a\u06d2 \u06c1\u06cc\u06ba \u0627\u0648\u0631 \u0627\u0646 \u06a9\u06d2 \u0644\u06cc\u06d2 \u0639\u0627\u0644\u0645 \u06a9\u06cc \u0631\u0627\u06c1\u0646\u0645\u0627\u0626\u06cc \u0636\u0631\u0648\u0631\u06cc \u06c1\u06d2\u06d4\n\n"
               "\u0628\u0631\u0627\u06c1\u0650 \u06a9\u0631\u0645 \u06a9\u0633\u06cc \u0645\u0633\u062a\u0646\u062f \u0639\u0627\u0644\u0650\u0645\u0650 \u062f\u06cc\u0646 \u0633\u06d2 \u0631\u062c\u0648\u0639 \u06a9\u0631\u06cc\u06ba\u060c \u06cc\u0627 {url} \u062c\u06cc\u0633\u06d2 \u0642\u0627\u0628\u0644\u0650 \u0627\u0639\u062a\u0645\u0627\u062f \u0630\u0631\u06cc\u0639\u06d2 \u067e\u0631 \u062f\u06cc\u06a9\u06be\u06cc\u06ba\u06d4"),
        "ar": ("\u0644\u0645 \u0623\u062c\u062f \u0625\u062c\u0627\u0628\u0629 \u0645\u0648\u062b\u0648\u0642\u0629 \u0648\u0645\u0633\u0646\u062f\u0629 \u0644\u0647\u0630\u0627 \u0627\u0644\u0633\u0624\u0627\u0644 \u0641\u064a \u0642\u0627\u0639\u062f\u0629 \u0645\u0639\u0631\u0641\u062a\u064a\u060c \u0648\u0644\u0646 \u0623\u062e\u0645\u0651\u0646\u060c \u0641\u0628\u0639\u0636 \u0627\u0644\u0645\u0633\u0627\u0626\u0644 \u062a\u0639\u062a\u0645\u062f \u0639\u0644\u0649 \u0638\u0631\u0648\u0641 \u0627\u0644\u0634\u062e\u0635 \u0648\u062a\u062a\u0637\u0644\u0628 \u0631\u0623\u064a \u0639\u0627\u0644\u0650\u0645.\n\n"
               "\u064a\u064f\u0631\u062c\u0649 \u0627\u0633\u062a\u0634\u0627\u0631\u0629 \u0639\u0627\u0644\u0650\u0645 \u0645\u0624\u0647\u0651\u0644\u060c \u0623\u0648 \u0627\u0644\u0631\u062c\u0648\u0639 \u0625\u0644\u0649 \u0645\u0635\u062f\u0631 \u0645\u0648\u062b\u0648\u0642 \u0645\u062b\u0644 {url}"),
        "roman": ("Mujhe apne knowledge base mein is sawal ka mustanad aur qabil-e-etimaad jawab nahi mila, "
                  "is liye main andaza nahi lagaunga. Kuch masail insaan ke halaat par munhasir hote hain "
                  "aur un ke liye aalim ki rahnumai zaroori hai.\n\n"
                  "Baraye meharbani kisi mustanad aalim se rujoo karein, ya {url} jaise qabil-e-etimaad zariye par dekhein."),
    },
    "greeting": {
        "en": "Wa Alaikum Assalam. How can I help you with your Islamic question today?",
        "ur": "\u0648\u0639\u0644\u06cc\u06a9\u0645 \u0627\u0644\u0633\u0644\u0627\u0645\u06d4 \u0645\u06cc\u06ba \u0622\u062c \u0622\u067e \u06a9\u06d2 \u06a9\u0633\u06cc \u062f\u06cc\u0646\u06cc \u0633\u0648\u0627\u0644 \u0645\u06cc\u06ba \u06a9\u06cc\u0633\u06d2 \u0645\u062f\u062f \u06a9\u0631 \u0633\u06a9\u062a\u0627 \u06c1\u0648\u06ba\u061f",
        "ar": "\u0648\u0639\u0644\u064a\u0643\u0645 \u0627\u0644\u0633\u0644\u0627\u0645. \u0643\u064a\u0641 \u064a\u0645\u0643\u0646\u0646\u064a \u0645\u0633\u0627\u0639\u062f\u062a\u0643 \u0641\u064a \u0633\u0624\u0627\u0644\u0643 \u0627\u0644\u0625\u0633\u0644\u0627\u0645\u064a \u0627\u0644\u064a\u0648\u0645\u061f",
        "roman": "Wa Alaikum Assalam. Main aaj aap ke kisi deeni sawal mein kaise madad kar sakta hoon?",
    },
    "gibberish": {
        "en": ("I couldn't find a clear Islamic question in that. Please type your question in a full "
               "sentence \u2014 for example, \u201cCan I combine prayers while travelling?\u201d \u2014 and I'll do my best to help."),
        "ur": ("\u0627\u0633 \u0645\u06cc\u06ba \u0645\u062c\u06be\u06d2 \u06a9\u0648\u0626\u06cc \u0648\u0627\u0636\u062d \u062f\u06cc\u0646\u06cc \u0633\u0648\u0627\u0644 \u0646\u06c1\u06cc\u06ba \u0645\u0644\u0627\u06d4 \u0628\u0631\u0627\u06c1\u0650 \u06a9\u0631\u0645 \u0627\u067e\u0646\u0627 \u0633\u0648\u0627\u0644 \u0645\u06a9\u0645\u0644 \u062c\u0645\u0644\u06d2 \u0645\u06cc\u06ba \u0644\u06a9\u06be\u06cc\u06ba \u2014 \u0645\u062b\u0644\u0627\u064b \u201c\u06a9\u06cc\u0627 \u0645\u06cc\u06ba \u0633\u0641\u0631 \u0645\u06cc\u06ba \u0646\u0645\u0627\u0632\u06cc\u06ba \u062c\u0645\u0639 \u06a9\u0631 \u0633\u06a9\u062a\u0627 \u06c1\u0648\u06ba\u061f\u201d \u2014 \u062a\u0627\u06a9\u06c1 \u0645\u06cc\u06ba \u0628\u06c1\u062a\u0631 \u0645\u062f\u062f \u06a9\u0631 \u0633\u06a9\u0648\u06ba\u06d4"),
        "ar": ("\u0644\u0645 \u0623\u062c\u062f \u0633\u0624\u0627\u0644\u0627\u064b \u0625\u0633\u0644\u0627\u0645\u064a\u0627\u064b \u0648\u0627\u0636\u062d\u0627\u064b \u0641\u064a \u0630\u0644\u0643. \u0627\u0644\u0631\u062c\u0627\u0621 \u0643\u062a\u0627\u0628\u0629 \u0633\u0624\u0627\u0644\u0643 \u0641\u064a \u062c\u0645\u0644\u0629 \u0643\u0627\u0645\u0644\u0629 \u2014 \u0645\u062b\u0644\u0627\u064b \u00ab\u0647\u0644 \u064a\u062c\u0648\u0632 \u0644\u064a \u0627\u0644\u062c\u0645\u0639 \u0628\u064a\u0646 \u0627\u0644\u0635\u0644\u0627\u062a\u064a\u0646 \u0623\u062b\u0646\u0627\u0621 \u0627\u0644\u0633\u0641\u0631\u061f\u00bb \u2014 \u0648\u0633\u0623\u0628\u0630\u0644 \u062c\u0647\u062f\u064a \u0644\u0644\u0645\u0633\u0627\u0639\u062f\u0629."),
        "roman": ("Is mein mujhe koi wazeh deeni sawal nahi mila. Baraye meharbani apna sawal mukammal "
                  "jumle mein likhein \u2014 masalan \u201cKya main safar mein namazein jama kar sakta hoon?\u201d \u2014 "
                  "taake main behtar madad kar sakoon."),
    },
    "ask_lang": {
        "en": "Sure \u2014 ask me your question and I'll answer in that language, in sha' Allah.",
        "ur": "\u0636\u0631\u0648\u0631\u06d4 \u0627\u067e\u0646\u0627 \u0633\u0648\u0627\u0644 \u067e\u0648\u0686\u06be\u06cc\u06ba\u060c \u0645\u06cc\u06ba \u0627\u0633\u06cc \u0632\u0628\u0627\u0646 \u0645\u06cc\u06ba \u062c\u0648\u0627\u0628 \u062f\u0648\u06ba \u06af\u0627\u060c \u0627\u0646 \u0634\u0627\u0621 \u0627\u0644\u0644\u06c1\u06d4",
        "ar": "\u062d\u0633\u0646\u0627\u064b \u2014 \u0627\u0637\u0631\u062d \u0633\u0624\u0627\u0644\u0643 \u0648\u0633\u0623\u062c\u064a\u0628\u064f\u0643 \u0628\u0647\u0630\u0647 \u0627\u0644\u0644\u063a\u0629\u060c \u0625\u0646 \u0634\u0627\u0621 \u0627\u0644\u0644\u0647.",
        "roman": "Zaroor \u2014 apna sawal poochein, main isi zabaan mein jawab doonga, in sha' Allah.",
    },
    "capability": {
        "en": ("You can ask me any Islamic Fiqh (jurisprudence) question \u2014 for example about "
               "prayer, fasting, zakat, hajj, marriage, or purification. I answer with verified "
               "Qur'an and Hadith citations across the Hanafi, Maliki, and Shafi'i schools. "
               "Go ahead and ask your question."),
        "ur": ("\u0622\u067e \u0645\u062c\u06be \u0633\u06d2 \u0641\u0642\u06c1 \u0633\u06d2 \u0645\u062a\u0639\u0644\u0642 \u06a9\u0648\u0626\u06cc \u0628\u06be\u06cc \u0633\u0648\u0627\u0644 \u067e\u0648\u0686\u06be \u0633\u06a9\u062a\u06d2 \u06c1\u06cc\u06ba \u2014 \u0645\u062b\u0644\u0627\u064b \u0646\u0645\u0627\u0632\u060c \u0631\u0648\u0632\u06c1\u060c \u0632\u06a9\u0648\u0629\u060c \u062d\u062c\u060c \u0646\u06a9\u0627\u062d \u06cc\u0627 \u0637\u06c1\u0627\u0631\u062a \u06a9\u06d2 \u0628\u0627\u0631\u06d2 \u0645\u06cc\u06ba\u06d4 \u0645\u06cc\u06ba \u062d\u0646\u0641\u06cc\u060c \u0645\u0627\u0644\u06a9\u06cc \u0627\u0648\u0631 \u0634\u0627\u0641\u0639\u06cc \u0645\u0633\u0627\u0644\u06a9 \u06a9\u06d2 \u0645\u0637\u0627\u0628\u0642 \u0645\u0633\u062a\u0646\u062f \u0642\u0631\u0622\u0646 \u0648 \u062d\u062f\u06cc\u062b \u06a9\u06d2 \u062d\u0648\u0627\u0644\u0648\u06ba \u06a9\u06d2 \u0633\u0627\u062a\u06be \u062c\u0648\u0627\u0628 \u062f\u06cc\u062a\u0627 \u06c1\u0648\u06ba\u06d4 \u0628\u0631\u0627\u06c1\u0650 \u06a9\u0631\u0645 \u0627\u067e\u0646\u0627 \u0633\u0648\u0627\u0644 \u067e\u0648\u0686\u06be\u06cc\u06ba\u06d4"),
        "ar": ("\u064a\u0645\u0643\u0646\u0643 \u0623\u0646 \u062a\u0633\u0623\u0644\u0646\u064a \u0623\u064a \u0633\u0624\u0627\u0644 \u0641\u064a \u0627\u0644\u0641\u0642\u0647 \u0627\u0644\u0625\u0633\u0644\u0627\u0645\u064a \u2014 \u0645\u062b\u0644\u0627\u064b \u0639\u0646 \u0627\u0644\u0635\u0644\u0627\u0629 \u0623\u0648 \u0627\u0644\u0635\u064a\u0627\u0645 \u0623\u0648 \u0627\u0644\u0632\u0643\u0627\u0629 \u0623\u0648 \u0627\u0644\u062d\u062c \u0623\u0648 \u0627\u0644\u0632\u0648\u0627\u062c \u0623\u0648 \u0627\u0644\u0637\u0647\u0627\u0631\u0629. \u0623\u064f\u062c\u064a\u0628 \u0628\u0627\u0644\u0627\u0633\u062a\u0646\u0627\u062f \u0625\u0644\u0649 \u0627\u0644\u0642\u0631\u0622\u0646 \u0648\u0627\u0644\u062d\u062f\u064a\u062b \u0627\u0644\u0645\u0648\u062b\u0651\u0642\u064a\u0646 \u0648\u0641\u0642\u064b\u0627 \u0644\u0644\u0645\u0630\u0627\u0647\u0628 \u0627\u0644\u062d\u0646\u0641\u064a \u0648\u0627\u0644\u0645\u0627\u0644\u0643\u064a \u0648\u0627\u0644\u0634\u0627\u0641\u0639\u064a. \u062a\u0641\u0636\u0651\u0644 \u0628\u0637\u0631\u062d \u0633\u0624\u0627\u0644\u0643."),
        "roman": ("Aap mujh se koi bhi Islamic Fiqh ka sawal pooch sakte hain \u2014 masalan namaz, roza, "
                  "zakat, hajj, nikah, ya taharat ke baare mein. Main Hanafi, Maliki aur Shafi'i "
                  "masalik ke mutabiq mustanad Qur'an aur Hadith ke hawalon ke saath jawab deta hoon. "
                  "Baraye meharbani apna sawal poochein."),
    },
}


def _msg(kind, query, **fmt):
    """Pick a fixed message in the user's detected language, with fallback to English."""
    table = _MSG.get(kind, {})
    text = table.get(_lang_key(query)) or table.get("en", "")
    return text.format(**fmt) if fmt else text


def _low_confidence_result(query, madhhab, pct, label):
    """
    Returned when confidence is below LOW_CONFIDENCE_PCT. We deliberately do not
    generate a ruling — making up an Islamic answer is prohibited for this
    project — and instead refer the user to a trusted scholarly resource.
    """
    msg = _msg("referral", query, url=ISLAMQA_URL)
    return {
        "ok": True, "mode": "low_confidence", "out_of_scope": False,
        "low_confidence": True,
        "answer": msg,
        "confidence": round(pct / 100, 3),
        "confidence_percent": pct, "confidence_label": label,
        "confidence_reason": (
            "The system's confidence for this question was too low to give a "
            "grounded ruling, so no answer was generated."
        ),
        "consensus": False,
        "referral_url": ISLAMQA_URL,
        "sources": {"fiqh": [], "quran": [], "hadith": []},
    }


def _confidence_reason(quran, hadith, top_topic_score, consensus):
    """One plain sentence explaining what drove the confidence score."""
    if   top_topic_score >= 0.85: match = "a very strong match to a curated topic"
    elif top_topic_score >= 0.65: match = "a strong match to a curated topic"
    elif top_topic_score >= 0.45: match = "a good match to a curated topic"
    else:                          match = "a partial match to a curated topic"

    if quran and hadith: backing = "backed by both Qur'an and hadith references"
    elif quran:          backing = "backed by a Qur'an reference"
    elif hadith:         backing = "backed by a hadith reference"
    else:                backing = "with limited reference backing"

    cons = ("and all major schools agree, which raises the score" if consensus
            else "but the schools differ on this, so it doesn't get the consensus boost")
    return f"This score reflects {match}, {backing}, {cons}."


def _is_vague_question(query):
    """True for short prompts that are grammatical but carry NO real subject to
    answer — e.g. 'what am I', 'who am I', 'help', 'tell me something', 'explain'.
    These have a question/instruction word but no topic noun, so the model just
    rambles ('please provide more details'). We route them to a rephrase prompt.
    Real short questions ('what is zakat', 'how many prayers') are NOT vague
    because they contain a topic word, so they pass through untouched.
    """
    raw = (query or "").strip()
    q = re.sub(r"[^\w\s]", " ", raw.lower()).strip()
    words = q.split()
    if not words:
        return False  # emptiness handled by _is_gibberish
    # Only consider SHORT prompts (long ones are genuine attempts).
    if len(words) > 5:
        return False
    # Script (Urdu/Arabic) input is handled elsewhere; only screen Latin here.
    if any('\u0600' <= ch <= '\u06FF' for ch in raw):
        return False

    # Known vague/contentless phrasings (exact, whole-message match).
    vague_exact = {
        "what am i", "who am i", "what is this", "what is it", "what", "who",
        "why", "how", "when", "where", "help", "help me", "tell me", "tell me something",
        "explain", "explain this", "explain it", "answer", "answer me", "info",
        "information", "something", "anything", "guide me", "assist me", "idk what to ask",
        "what should i ask", "what can you do", "what do you do", "who are you",
        "what are you", "define", "meaning", "kuch batao", "kuch bta", "batao",
    }
    if q in vague_exact:
        return True

    # General case: has a question/instruction word but no recognisable TOPIC.
    lead_words = {
        "what", "who", "why", "how", "when", "where", "which", "am", "is", "are",
        "tell", "explain", "help", "define", "give", "show", "answer", "describe",
    }
    if words[0] not in lead_words:
        return False
    # Reuse the domain vocabulary: if ANY token looks like a topic/subject
    # (a noun that isn't a pure function word), it's a real question.
    function_words = {
        "what", "who", "why", "how", "when", "where", "which", "am", "is", "are",
        "i", "me", "my", "you", "your", "it", "this", "that", "the", "a", "an",
        "do", "does", "can", "to", "of", "in", "on", "for", "about", "tell",
        "explain", "help", "give", "show", "answer", "describe", "define", "should",
        "would", "will", "and", "or", "please", "some", "any", "thing", "something",
        "anything", "us", "we", "them",
    }
    content_tokens = [w for w in words if w not in function_words and len(w) >= 3]
    # No content noun at all -> vague.
    return len(content_tokens) == 0


def _is_gibberish(query):
    """True when the message carries no real question — random tokens in ANY
    language ('yo', 'blah', 'asdf', Urdu 'کہل', Arabic scribble, Roman-Urdu
    'abcd'), single filler words, or no letters at all. Such input must NOT be
    pushed through retrieval (it grabs a random topic and returns a bogus cited
    answer). It is routed to a rephrase prompt instead.

    Strategy: instead of trying to list every possible junk word (impossible
    across four languages), we require SHORT messages to contain at least one
    recognisable signal — a question word or an Islamic-domain word in English,
    Roman Urdu, Urdu script, or Arabic. Long, sentence-like input is assumed to
    be a genuine attempt and passes through.
    """
    raw = (query or "").strip()
    q = re.sub(r"[^\w\s]", " ", raw.lower()).strip()
    words = q.split()
    if not words:
        return True
    # No letters at all (pure numbers/symbols/emoji).
    if not any(ch.isalpha() for ch in q):
        return True

    # Obvious English/Latin filler.
    filler = {
        "yo", "blah", "blahblah", "bla", "asdf", "asdff", "qwerty", "test", "testing",
        "bruh", "bro", "lol", "lmao", "hmm", "hmmm", "meh", "idk", "uh", "um", "erm",
        "abc", "xyz", "aaa", "aaaa", "haha", "hahaha", "nvm", "wtf", "yeah", "yep",
        "nope", "nah", "yup", "sup", "wassup", "wagwan", "huh", "eh", "ok", "okay",
        "bla bla", "blabla", "kk", "k", "hmmmm", "ugh", "oof", "eh", "hey",
    }
    if all(w in filler for w in words):
        return True

    # Repeated single-character run in one token ("aaaaaa", "ھھھھ").
    if len(words) == 1 and len(set(words[0])) <= 1:
        return True

    # --- Cross-language "is there any real signal?" check for SHORT input ---
    # Long input (4+ words) is treated as a genuine attempt and passes through;
    # retrieval + downstream confidence gating handle the rest.
    if len(words) >= 4:
        return False

    # Recognisable signal words across the four supported languages. A short
    # message with NONE of these is almost certainly a random token.
    question_words = {
        # English
        "how", "what", "when", "why", "where", "who", "can", "is", "are", "should",
        "does", "do", "may", "will", "would", "which", "many",
        # Roman Urdu
        "kya", "kaise", "kaisay", "kab", "kahan", "kahaan", "kyun", "kyu", "kiya",
        "kia", "sakta", "sakte", "sakti", "hai", "hain", "ho", "kar", "karna", "krna",
        "kitne", "kitni", "kitna", "kaun", "konsa", "konsi", "kon", "kaunsa",
        "parhna", "parhni", "parhne", "farz", "wajib", "sunnat", "daily", "rozana",
        # (Urdu-script & Arabic question words handled by the domain set below too)
    }
    # Islamic-domain words (roots) in Latin letters (English + Roman Urdu).
    domain_latin = {
        "namaz", "namaaz", "naamaz", "namaz", "namz", "salah", "salat", "salaah",
        "prayer", "prayers", "pray", "wudu", "wuzu", "wudhu", "ablution",
        "roza", "roze", "rozay", "fast", "fasting", "sawm", "saum", "zakat", "zakah",
        "zakaat", "hajj", "haj", "umrah", "umra", "umrah",
        "quran", "qur'an", "koran", "hadith", "hadees", "sunnah", "sunnat", "halal",
        "haram", "haraam", "jaiz", "najaiz", "fiqh", "deen", "deeni", "ibadat", "ibadah",
        "ramadan", "ramzan", "eid", "janaza", "janazah", "nikah", "talaq", "witr",
        "sujood", "sajda", "sajdah", "ruku", "tayammum", "ghusl", "iddah", "mahr",
        "fitr", "qibla", "qiblah", "masjid", "dua", "duaa", "tasbih", "islam",
        "islamic", "muslim", "allah", "nabi", "rasool", "makkah", "madina", "kaaba",
        "tahajjud", "fajr", "dhuhr", "zuhr", "zohar", "asr", "maghrib", "isha",
        "rakat", "rakah", "rakats", "rakaat", "rakaats", "rakah", "interest", "riba",
        "sood", "sood", "qaza", "qada", "farz", "fardh", "wajib", "sunnah", "nafl",
        "namazein", "namazen", "prayers",
    }
    # Urdu/Arabic-script domain & question words (substring match on the raw text).
    domain_script = [
        "\u0646\u0645\u0627\u0632",  # namaz
        "\u0631\u0648\u0632",        # roza
        "\u0648\u0636\u0648", "\u0648\u0632\u0648",  # wudu
        "\u0632\u06a9\u0627", "\u0632\u06a9\u0648\u0629",  # zakat
        "\u062d\u062c",              # hajj
        "\u0639\u0645\u0631\u06c1", "\u0639\u0645\u0631\u0629",  # umrah
        "\u0642\u0631\u0622\u0646", "\u0642\u0631\u0627\u0646",  # quran
        "\u062d\u062f\u06cc\u062b",  # hadith
        "\u062d\u0644\u0627\u0644", "\u062d\u0631\u0627\u0645",  # halal/haram
        "\u0631\u0648\u0632\u06c1", "\u0635\u0648\u0645",  # fasting/sawm
        "\u0646\u06a9\u0627\u062d", "\u0637\u0644\u0627\u0642",  # nikah/talaq
        "\u063a\u0633\u0644", "\u062f\u0639\u0627",  # ghusl/dua
        "\u0627\u0633\u0644\u0627\u0645", "\u0627\u0644\u0644\u06c1", "\u0627\u0644\u0644\u0647",  # islam/allah
        "\u0645\u0633\u062c\u062f", "\u0639\u06cc\u062f", "\u0631\u0645\u0636\u0627\u0646",  # masjid/eid/ramadan
        # question words (Urdu/Arabic)
        "\u06a9\u06cc\u0627", "\u06a9\u06cc\u0633\u06d2", "\u06a9\u0628",  # kya/kaise/kab (Urdu)
        "\u0647\u0644", "\u0645\u0627\u0630\u0627", "\u0643\u064a\u0641", "\u0645\u062a\u0649", "\u0644\u0645\u0627\u0630\u0627", "\u0623\u064a\u0646",  # Arabic Q words
        "\u064a\u062c\u0648\u0632", "\u0635\u0644\u0627\u0629", "\u0635\u0648\u0645", "\u0632\u0643\u0627\u0629",  # yajuz/salah/sawm/zakah
    ]

    latin_has_signal = any(w in question_words or w in domain_latin for w in words)
    script_has_signal = any(tok in raw for tok in domain_script)
    if latin_has_signal or script_has_signal:
        return False

    # Short, and no recognisable question/domain signal in any language -> junk.
    return True


def _force_general(query):
    """Basic-fact questions that are never specific edge-case rulings -> general mode."""
    q = query.lower()
    # "how many ..." is always a basic count question (prayers, rakats, pillars...).
    if "how many" in q:
        return True
    return False


def _is_meta_instruction(query):
    """Bare language/format instructions ('answer in roman urdu', 'in urdu',
    'translate this') aren't questions — they mean 're-express the previous
    answer'. We accept short phrases that reference a language or a
    translate/explain-again instruction, as long as they carry no real question."""
    q = re.sub(r"[^\w\s]", "", query.lower()).strip()
    words = q.split()
    if len(words) > 8:
        return False
    # mentions a language/answering instruction but contains no actual question
    lang_words = ["roman urdu", "urdu", "english", "arabic", "hindi", "language"]
    instruct_words = ["answer", "reply", "respond", "speak", "talk", "write", "say it",
                      "translate", "explain", "rephrase", "convert", "make it", "say that",
                      "in", "to"]
    has_lang = any(w in q for w in lang_words)
    has_instruct = any(w in q for w in instruct_words)
    # A standalone translate/repeat request with no language named still counts
    # (e.g. "translate this", "say that again", "rephrase").
    translate_only = any(w in q for w in
        ["translate", "rephrase", "reword", "say that again", "repeat that", "again"])
    has_question = "?" in query or any(w in q for w in
        ["how", "what", "when", "why", "can i", "is it", "should", "kya", "kaise", "kab"])
    # Guard: a bare "in"/"to" alone shouldn't trigger — require it to co-occur
    # with a language word so plain sentences don't get misread.
    if not has_lang and not translate_only:
        return False
    return (has_lang or translate_only) and has_instruct and not has_question


def _is_capability_question(query):
    """True when the user is asking what the bot can do / what they may ask, e.g.
    'what can I ask', 'what can you do', 'what topics do you cover', and the
    Roman-Urdu / Urdu / Arabic equivalents ('mai kya pooch sakta hu',
    'tum kya karte ho', 'ماذا يمكنني أن أسأل'). These are NOT Fiqh questions, so
    retrieval must not run — we reply with a short capability message instead.

    Guarded to whole/short messages so a real question that merely contains one
    of these words (e.g. 'what can I ask Allah for in dua?') still gets answered.
    """
    raw = (query or "").strip()
    q = re.sub(r"[^\w\s]", " ", raw.lower()).strip()
    q = re.sub(r"\s+", " ", q)
    if not q:
        return False
    words = q.split()
    if len(words) > 7:                     # capability questions are short
        return False

    # 1) Latin (English + Roman Urdu) — match the WHOLE message against a set of
    #    known capability phrasings, plus a couple of tight patterns.
    latin_exact = {
        # English
        "what can i ask", "what can i ask you", "what can you do", "what do you do",
        "what can you help with", "what can you help me with", "what should i ask",
        "what can i ask about", "what topics can i ask", "what topics do you cover",
        "what kind of questions can i ask", "what questions can i ask",
        "what are you for", "what is this for", "what can this do", "how can you help",
        "how can you help me", "what all can i ask", "what can u ask", "what can u do",
        # Roman Urdu
        "mai kya pooch sakta hu", "main kya pooch sakta hu", "mai kya puch sakta hu",
        "mai kya pooch sakta hoon", "main kya pooch sakta hoon", "kya pooch sakta hu",
        "kya pooch sakta hoon", "mai aap se kya pooch sakta hu",
        "main aap se kya pooch sakta hoon", "aap kya karte ho", "tum kya karte ho",
        "tum kya kar sakte ho", "aap kya kar sakte ho", "mujhe kya poochna chahiye",
        "mai kya poochun", "main kya poochun", "kya poochun", "aap kis cheez mein madad karte ho",
        "mai kya sawal pooch sakta hu", "main kya sawal pooch sakta hoon",
    }
    if q in latin_exact:
        return True
    # Tight pattern: "(what/kya) ... (ask/pooch/puch) ..." within a short message,
    # e.g. "what can i ask here", "mai tumse kya pooch sakta hu".
    if len(words) <= 7:
        has_q = any(w in ("what", "kya", "kia") for w in words)
        has_ask = any(w in ("ask", "pooch", "poochun", "puch", "poch", "poochna",
                            "sawal", "questions", "question") for w in words)
        has_cap = any(w in ("can", "sakta", "sakte", "sakti", "chahiye", "karte",
                            "karsakta", "kar") for w in words)
        if has_q and has_ask and has_cap:
            return True

    # 2) Urdu / Arabic script — key phrases.
    #    Urdu:  "میں کیا پوچھ سکتا ہوں", "آپ کیا کر سکتے ہیں", "کیا پوچھوں"
    #    Arabic:"ماذا يمكنني أن أسأل", "ماذا تستطيع أن تفعل", "بماذا يمكنك مساعدتي"
    script_markers = [
        "\u06a9\u06cc\u0627 \u067e\u0648\u0686\u06be",      # kya poochh (Urdu)
        "\u06a9\u06cc\u0627 \u0633\u0648\u0627\u0644",      # kya sawal (Urdu)
        "\u06a9\u06cc\u0627 \u06a9\u0631 \u0633\u06a9\u062a\u06d2",  # kya kar sakte (Urdu)
        "\u0645\u0627\u0630\u0627 \u064a\u0645\u0643\u0646\u0646\u064a",  # matha yumkinuni (Arabic: what can I)
        "\u0645\u0627\u0630\u0627 \u0623\u0633\u0623\u0644",  # matha as'al (Arabic: what do I ask)
        "\u0645\u0627\u0630\u0627 \u062a\u0633\u062a\u0637\u064a\u0639",  # matha tastati' (Arabic: what can you)
        "\u0628\u0645\u0627\u0630\u0627 \u064a\u0645\u0643\u0646\u0643",  # bimatha yumkinuk (Arabic: with what can you)
    ]
    if len(raw) <= 60 and any(m in raw for m in script_markers):
        return True

    return False


def _is_greeting(query):
    """True when the message is ONLY a greeting / pleasantry, not a question.

    Catches English, Roman-Urdu and Urdu-script greetings. If a greeting is
    followed by a real question (e.g. "salam, kya main safar mein namaz jama kar
    sakta hu?"), we do NOT treat it as a greeting — the question wins and goes to
    retrieval. Matching is phrase-based (whole message or clear prefix) rather
    than loose substring, so a greeting word inside a longer question can't
    hijack it.
    """
    raw = (query or "").strip()
    q = raw.lower().strip()
    q_clean = re.sub(r"[^\w\s]", "", q)            # drop punctuation
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    if not q_clean:
        return False
    words = q_clean.split()
    if len(words) > 6:                             # long messages aren't bare greetings
        return False

    # Whole-message / prefix greeting phrases (English + Roman Urdu).
    greetings = [
        # salutations
        "assalam", "assalamu", "assalam o alaikum", "assalam alaikum",
        "salam", "salaam", "asalam", "asalaam", "asslam", "as salam",
        "walaikum", "wa alaikum", "walekum", "walaikum assalam",
        # English greetings / pleasantries
        "hi", "hello", "hey", "hola", "yo",
        "good morning", "good evening", "good afternoon", "good night",
        "how are you", "how are you doing", "hows it going", "how do you do",
        "thanks", "thank you", "shukran", "jazakallah", "jazak allah", "jazakillah",
        "ok", "okay", "test", "testing",
        # Roman-Urdu greetings / pleasantries
        "kaise ho", "kaise hain", "kaisay ho", "kaisay hain",
        "kaisa hai", "kaise ho bhai", "kaise ho aap", "aap kaise ho", "aap kaise hain",
        "kya haal hai", "kya haal", "kya haal hain", "kya hal hai", "kya hal",
        "sab theek", "sab thik", "sab theek hai", "sab thik hai", "sab khairiyat",
        "theek ho", "thik ho", "theek hain", "khairiyat",
        "kaise ho tum", "kese ho", "kesi ho",
    ]

    # 1) If the WHOLE message (punctuation stripped) is exactly a greeting phrase,
    #    it's a greeting even with a trailing "?" ("kaise ho?").
    if q_clean in greetings:
        return True

    # 2) Otherwise, a "?" or a real question signal means send it to retrieval,
    #    even if it opens with a greeting. (Islamic/topic words are NOT treated as
    #    question signals here — only interrogative structure.)
    if "?" in raw:
        return False
    question_markers = [
        "how many", "how do", "how can i", "what is", "what are", "when", "why",
        "can i", "is it", "should i", "how to", "which",
        # Roman-Urdu interrogatives that indicate a real question follows
        "kya main", "kya mai", "kaise parh", "kaise karu", "kaise karun",
        "kitni", "kitne", "kitna", "kab", "kahan", "kaha", "jaiz hai", "haram hai",
        "kar sakta", "kar sakti", "karna chahiye", "parh sakta",
    ]
    if any(m in q for m in question_markers):
        return False

    # 3) Greeting at the start, remainder isn't a real question -> still a greeting
    #    (e.g. "salam bhai", "assalam o alaikum sheikh").
    for g in greetings:
        if q_clean.startswith(g + " "):
            return True
    return False


_TARGET_LANG_WORDS = {
    "urdu":       "Urdu script (اردو)",
    "roman urdu": "Roman Urdu (Urdu written in Latin letters)",
    "roman":      "Roman Urdu (Urdu written in Latin letters)",
    "arabic":     "Arabic",
    "english":    "English",
}


# Roman-Urdu marker words (Urdu written in Latin letters). Used to tell apart
# English from Roman Urdu, since both use the Latin alphabet.
_ROMAN_URDU_HINTS = {
    # ONLY grammatical/function words that are distinctly Urdu. Islamic terms
    # (wudu, namaz, roza, halal, zakat, hajj...) are deliberately EXCLUDED
    # because English speakers use them too ("is coffee halal", "vomiting and
    # wudu"), which caused English questions to be misread as Roman Urdu.
    "kya", "kaise", "kaisay", "kab", "kahan", "kahaan", "kyun", "kyu", "kiya", "kia",
    "hai", "hain", "ho", "hota", "hoti", "hote", "tha", "thi", "the",
    "karna", "karne", "karta", "karti", "karte", "kar", "krna", "krta", "krte", "kr",
    "sakta", "sakte", "sakti", "krsakta", "krskta",
    "mai", "main", "mein", "meri", "mera", "mere", "mujhe", "mujhko", "hum", "humein",
    "ap", "aap", "tum", "tumhe", "uska", "uski", "unka", "iska", "iski",
    "nhi", "nahi", "nahin", "haan", "acha", "achha", "theek", "thik",
    "batao", "bta", "bata", "batayen", "bataye", "bataiye",
    "chahiye", "chahie", "chaida", "hoga", "hogi", "hoga", "raha", "rahi", "rahe",
    "kaha", "kehte", "kehna", "wala", "wali", "wale", "agar", "lekin", "aur",
    "par", "se", "ka", "ke", "ki", "ko", "ne", "kis", "kaun", "kitna", "kitni",
}


def _detect_language(query):
    """Deterministically detect the input language so we can force the reply to
    match it. Returns one of: 'Urdu script (اردو)', 'Arabic', 'Roman Urdu ...',
    or 'English'. Script-based checks are reliable; Roman-Urdu vs English is a
    heuristic on common Urdu words written in Latin letters."""
    q = query or ""
    # Arabic/Urdu share the Arabic block (U+0600–U+06FF). Distinguish by the
    # Urdu-specific letters that Arabic does not use (ں ہ ے ٹ ڈ ڑ گ چ پ ژ ک).
    has_arabic_block = any('\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F'
                           or '\uFB50' <= ch <= '\uFDFF' or '\uFE70' <= ch <= '\uFEFF'
                           for ch in q)
    if has_arabic_block:
        urdu_specific = set("ںہھیےۓٹڈڑگچپژکﺁآۃ")
        if any(ch in urdu_specific for ch in q):
            return _TARGET_LANG_WORDS["urdu"]
        # Heuristic: if it also has clearly Urdu function words in script, call Urdu.
        return _TARGET_LANG_WORDS["arabic"]
    # Latin script: decide English vs Roman Urdu by marker words.
    words = re.sub(r"[^\w\s]", " ", q.lower()).split()
    if words:
        hits = sum(1 for w in words if w in _ROMAN_URDU_HINTS)
        # Require a genuine signal: at least 2 Urdu function words, OR one strong
        # one in a short phrase where it dominates. A single ambiguous short word
        # ('se', 'ka') in an otherwise-English sentence must NOT flip it to Urdu.
        strong = {"kya", "kaise", "kaisay", "kyun", "kyu", "hai", "hain", "karna",
                  "krna", "sakta", "sakte", "mujhe", "mein", "nahi", "nahin",
                  "batao", "bata", "chahiye", "kitna", "kitni", "kaun", "kahan"}
        strong_hits = sum(1 for w in words if w in strong)
        if strong_hits >= 1 and (hits / len(words)) >= 0.25:
            return _TARGET_LANG_WORDS["roman urdu"]
        if hits >= 2 and (hits / len(words)) >= 0.4:
            return _TARGET_LANG_WORDS["roman urdu"]
    return _TARGET_LANG_WORDS["english"]


def _requested_target_language(query):
    """Return a human-readable target language named in a meta-instruction, or None."""
    q = query.lower()
    # check the more specific 'roman urdu' before 'urdu'
    for key in ("roman urdu", "roman", "urdu", "arabic", "english"):
        if key in q:
            return _TARGET_LANG_WORDS[key]
    return None


def _reexpress_result(prev_answer, instruction, madhhab=None):
    """
    Re-express an EXISTING answer in a requested language (e.g. "explain in urdu"
    as a follow-up). No retrieval, no classifier — we only translate/rephrase the
    text we already produced, preserving any [citations] exactly as-is.
    """
    target = _requested_target_language(instruction) or "the requested language"
    prompt = f"""Re-express the following Islamic answer in {target}.

RULES:
- Keep the meaning EXACTLY the same. Do not add new rulings, verses, or facts.
- Keep every bracketed citation (e.g. [Quran 2:196], [Sahih Bukhari #8]) EXACTLY as written, in Latin letters, unchanged.
- If the target is Urdu script, write fluent natural Urdu (اردو). You ARE capable of writing Urdu — never claim otherwise and never substitute English.
- If the target is Roman Urdu, write Urdu using Latin letters.
- Never use Hindi or Devanagari script.
- Output only the re-expressed answer, nothing else.

ANSWER TO RE-EXPRESS:
{prev_answer}
"""
    out = gemini_generate(prompt)
    return {
        "ok": True, "mode": "reexpress", "out_of_scope": False,
        "answer": out.strip(),
        # This is the same content as before, just re-expressed — no new scoring.
        "confidence": None, "confidence_percent": None, "confidence_label": None,
        "consensus": False,
        "sources": {"fiqh": [], "quran": [], "hadith": []},
    }


def answer_question(query, madhhab=None, prev_answer=None):
    """
    Full pipeline. Returns a JSON-serialisable dict:
      { ok, mode, answer, confidence, confidence_percent, confidence_label,
        consensus, sources: {fiqh:[...], quran:[...], hadith:[...]}, out_of_scope }

    prev_answer: the previous bot answer (optional). When the current message is
    a bare "explain/translate in <language>" instruction, we re-express THIS text
    in the requested language instead of running retrieval (which would otherwise
    grab a random nearby topic).
    """
    if not _READY:
        init(verbose=False)
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "empty_query"}

    # ── Greeting / small-talk gate ───────────────────────────────
    # A greeting is not a question — don't run retrieval and answer something
    # that was never asked. Reply with a short greeting + a prompt to ask.
    if _is_greeting(query):
        return {
            "ok": True, "mode": "greeting", "out_of_scope": False,
            "answer": _msg("greeting", query),
            "confidence": None, "confidence_percent": None, "confidence_label": None,
            "consensus": False, "sources": {"fiqh": [], "quran": [], "hadith": []},
        }

    # ── Capability / "what can I ask" gate ───────────────────────
    # Questions about what the bot can do ("what can I ask", "mai kya pooch sakta
    # hu", "ماذا يمكنني أن أسأل") are NOT Fiqh questions. Without this, retrieval
    # grabs a nearby topic and returns a random cited ruling. Reply with a short
    # capability message in the user's language instead.
    if _is_capability_question(query):
        return {
            "ok": True, "mode": "capability", "out_of_scope": False,
            "answer": _msg("capability", query),
            "confidence": None, "confidence_percent": None, "confidence_label": None,
            "consensus": False, "sources": {"fiqh": [], "quran": [], "hadith": []},
        }

    # Bare language/format instruction (e.g. "answer me in roman urdu") — not a
    # question. If we have the previous answer, RE-EXPRESS it in the requested
    # language (that's what "explain in urdu" as a follow-up means). Otherwise
    # just acknowledge and wait for the actual question.
    if _is_meta_instruction(query):
        if prev_answer and prev_answer.strip():
            return _reexpress_result(prev_answer, query, madhhab)
        return {
            "ok": True, "mode": "greeting", "out_of_scope": False,
            "answer": _msg("ask_lang", query),
            "confidence": None, "confidence_percent": None, "confidence_label": None,
            "consensus": False, "sources": {"fiqh": [], "quran": [], "hadith": []},
        }

    topic_matches = classify_topic(query, top_n=2)
    top_topic_id    = topic_matches[0][0] if topic_matches else None
    top_topic_score = topic_matches[0][1] if topic_matches else 0.0

    # ── Gibberish / no-real-question gate ────────────────────────
    # Random tokens ("yo", "blah", "asdf") are not questions. Retrieval would
    # still grab a nearest topic and fabricate a cited answer, so we intercept
    # junk input and return an honest low-confidence prompt to rephrase.
    if _is_gibberish(query):
        return {
            "ok": True, "mode": "low_confidence", "out_of_scope": True,
            "answer": _msg("gibberish", query),
            "confidence": None, "confidence_percent": None, "confidence_label": "Low",
            "consensus": False, "sources": {"fiqh": [], "quran": [], "hadith": []},
        }

    # ── Vague / contentless question gate ────────────────────────
    # Grammatical but empty prompts ("what am I", "who am I", "help", "explain")
    # have a question word but no topic, so the model just rambles. Ask the user
    # to be specific instead of guessing.
    if _is_vague_question(query):
        return {
            "ok": True, "mode": "low_confidence", "out_of_scope": True,
            "answer": _msg("gibberish", query),
            "confidence": None, "confidence_percent": None, "confidence_label": "Low",
            "consensus": False, "sources": {"fiqh": [], "quran": [], "hadith": []},
        }
    # Grave, contested, or highly personal matters (violence, sexuality,
    # apostasy, abortion, divorce specifics, etc.) are ALWAYS referred to a
    # qualified scholar. A small automated system must not issue a ruling on
    # these, even if a curated topic weakly matches — the risk of a wrong or
    # oversimplified answer is too high.
    if _is_sensitive_question(query):
        pct, label = confidence_to_percent(0.30)
        return _low_confidence_result(query, madhhab, pct, label)

    # ── General-knowledge routing (decided up front) ─────────────
    # Basic-fact questions ("how many prayers?") and anything that doesn't clearly
    # match a curated fiqh topic are answered from general Islamic knowledge rather
    # than forced through a grounded ruling that may grasp at the wrong topic.
    if _force_general(query) or top_topic_score < GENERAL_IF_BELOW:
        return _general_result(query, madhhab, top_topic_id, top_topic_score)

    fiqh = retrieve_fiqh_ruling(query, madhhab=madhhab, top_k=2)

    # ── Classifier anchoring ─────────────────────────────────────
    # The fine-tuned topic classifier (~97% accurate) is the source of truth for
    # WHICH topic a question belongs to. Pure hybrid retrieval on topic text can
    # be fooled by near-identical wudu/fasting topics (e.g. vomiting vs bleeding).
    # When the classifier confidently disagrees with the hybrid top hit, promote
    # the classifier's topic so the ruling, references and sources all align.
    CLASSIFIER_TRUST = 0.50
    if (top_topic_id and top_topic_score >= CLASSIFIER_TRUST
            and (not fiqh or fiqh[0]["topic_id"] != top_topic_id)):
        anchored = _fiqh_entry_by_id(top_topic_id, madhhab, score=top_topic_score)
        if anchored:
            fiqh = [anchored] + [r for r in fiqh if r["topic_id"] != top_topic_id]
            fiqh = fiqh[:2]

    quran_refs, hadith_refs = [], []
    if fiqh:
        refs = fiqh[0].get("references", {}) or {}
        quran_refs  = refs.get("quran", []) or []
        hadith_refs = refs.get("hadith", []) or []

    quran = retrieve_quran(query, fiqh_refs=quran_refs, top_k=2)
    # Curated/direct ayahs score 1.0 and always pass; gate weak fallback matches.
    quran = [v for v in quran if v["score"] >= QURAN_MIN_SCORE]

    hadith = []
    for href in hadith_refs:
        src = normalize_hadith_source(href.get("source", ""))
        hno = href.get("hadith_no", "")
        h = hadith_lookup.get((src, str(hno)))
        if h:
            # curated reference present in the local corpus (full text available)
            hadith.append({"type": "hadith", "source": h["source"], "hadith_no": h["hadith_no"],
                           "chapter": h.get("chapter", ""), "text_en": h.get("text_en", ""),
                           "text_ar": h.get("text_ar", ""), "score": 1.0, "curated": True})
        # References not found in either the Bukhari/Muslim corpus or the
        # supplementary other_hadith.json are skipped (we only cite hadith whose
        # text we can actually display).
    if not hadith and ENABLE_HADITH_SEARCH:
        # Opt-in only. Guessed (search) hadith are gated hard and never trusted by default.
        hadith = [h for h in retrieve_hadith(query, top_k=2) if h["score"] >= HADITH_MIN_SCORE]

    topic_gap = (topic_matches[0][1] - topic_matches[1][1]) if len(topic_matches) > 1 else top_topic_score
    confidence = compute_confidence(fiqh, hadith, quran, topic_score=top_topic_score, topic_gap=topic_gap)
    pct, label = confidence_to_percent(confidence)

    # ── Low-confidence guard ─────────────────────────────────────
    # If the system's own confidence is Low (< LOW_CONFIDENCE_PCT), we do NOT
    # generate a ruling. Fabricating or guessing an Islamic answer is prohibited
    # for this project, so we decline honestly and point the user to a trusted
    # scholarly resource instead of returning a shaky answer.
    if pct < LOW_CONFIDENCE_PCT:
        return _low_confidence_result(query, madhhab, pct, label)

    prompt = build_rag_prompt(query, madhhab, fiqh, hadith, quran,
                              target_language=_detect_language(query))
    answer = gemini_generate(prompt)
    answer = _collapse_topic_tags(answer)   # drop repeated fiqh-topic tags

    # If the grounded path couldn't answer (model returned the scholar-referral
    # line), fall back to a general-knowledge answer instead of showing a refusal.
    if _is_refusal(answer):
        return _general_result(query, madhhab, top_topic_id, top_topic_score)

    def slim_fiqh(r):
        return {"topic_id": r["topic_id"], "topic_name": r["topic_name"],
                "consensus": r["consensus"], "scholar_review": r.get("scholar_review"),
                "score": round(r["score"], 3)}
    def slim_quran(v):
        return {"ref": f'{v["chapter"]}:{v["ayah"]}', "surah": v.get("surah_name_en", ""),
                "arabic": v.get("arabic", ""), "english": v.get("english", ""),
                "urdu": v.get("urdu", "")}
    def slim_hadith(h):
        return {"ref": f'{h["source"]} #{h["hadith_no"]}', "chapter": h.get("chapter", ""),
                "text_ar": h.get("text_ar", ""), "text_en": h.get("text_en", "")}

    return {
        "ok": True, "mode": "fiqh", "out_of_scope": False,
        "answer": answer,
        "confidence": confidence, "confidence_percent": pct, "confidence_label": label,
        "confidence_reason": _confidence_reason(
            quran, hadith, top_topic_score, bool(fiqh[0].get("consensus")) if fiqh else False),
        "consensus": bool(fiqh[0].get("consensus")) if fiqh else False,
        "sources": {
            "fiqh":   [slim_fiqh(r)  for r in fiqh],
            "quran":  [slim_quran(v) for v in quran],
            "hadith": [slim_hadith(h) for h in hadith],
        },
    }


if __name__ == "__main__":
    init()
    import pprint
    pprint.pprint(answer_question("Can I combine Zuhr and Asr while travelling?", "Hanafi"))
