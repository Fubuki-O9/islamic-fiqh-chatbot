const CONFIG = {
    // The Gemini key now lives ONLY in the backend (.env) — never in the browser.
    API_KEY: "",

    // Address of your Flask backend (rag_engine + Gemini + Mongo logging).
    // Change this only if you host the backend somewhere other than this machine.
    BACKEND_URL: "http://localhost:5000"
};
