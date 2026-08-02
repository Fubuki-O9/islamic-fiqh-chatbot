document.addEventListener('DOMContentLoaded', () => {

    // ── Backend URL (needed early for verse fetch) ──────────────
    const BACKEND_URL = (typeof CONFIG !== 'undefined' && CONFIG.BACKEND_URL)
        ? CONFIG.BACKEND_URL.replace(/\/+$/, '')
        : 'http://localhost:5000';

    // ── Loading Screen ──────────────────────────────────────────
    const loadingScreen = document.getElementById('loading-screen');
    const verseArabic = document.querySelector('.loader-verse-arabic');
    const verseTrans = document.querySelector('.loader-verse-translation');
    const verseRef = document.querySelector('.loader-verse-ref');

    // Fetch a random verse from the backend
    (async function loadVerse() {
        if (!verseArabic) return;
        try {
            const resp = await fetch(`${BACKEND_URL}/verse`);
            if (resp.ok) {
                const v = await resp.json();
                verseArabic.textContent = v.arabic;
                verseTrans.textContent = `"${v.english}"`;
                const surahAr = v.surah_ar || '';
                verseRef.textContent = `${surahAr} ${v.chapter}:${v.ayah}`;
            }
        } catch (e) { console.warn("Could not load verse", e); }
    })();

    // ── Utility: Detect Urdu/Arabic text ────────────────────────
    function isUrduText(text) {
        if (!text) return false;
        return /[\u0600-\u06FF\u0750-\u077F]/.test(text);
    }

    const loadStart = Date.now();
    const enterBtn = document.getElementById('loader-enter-btn');
    const loaderBar = document.getElementById('loader-bar-track');

    const showEnterBtn = () => {
        const elapsed = Date.now() - loadStart;
        const delay = Math.max(0, 2800 - elapsed);
        setTimeout(() => {
            enterBtn.classList.add('show');
            loaderBar.style.opacity = '0';
            loaderBar.style.height = '0';
            loaderBar.style.margin = '0';
        }, delay);
    };

    enterBtn.addEventListener('click', () => {
        loadingScreen.classList.add('hide');
    });

    // ── DOM Elements ─────────────────────────────────────────────
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const sttStatus = document.getElementById('stt-status');
    const sttStatusText = document.getElementById('stt-status-text');
    const sttCancelBtn = document.getElementById('stt-cancel-btn');
    const madhabSelection = document.getElementById('madhab-selection');
    const madhabCards = document.querySelectorAll('.madhab-card');
    const sidebarMadhabName = document.getElementById('sidebar-madhab-name');
    const changeMadhabBtn = document.getElementById('change-madhab-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const historyList = document.getElementById('history-list');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeToggleIcon = document.getElementById('theme-toggle-icon');
    const themeToggleText = document.getElementById('theme-toggle-text');

    // ── State ────────────────────────────────────────────────────
    let chats = JSON.parse(localStorage.getItem('islamic_chatbot_chats')) || [];
    let currentChatId = localStorage.getItem('islamic_chatbot_current_id') || null;
    let apiKey = '';
    let isRecording = false;
    let recognition = null;
    let currentAudio = null; // track server TTS audio so we can stop it

    // Load API key from config.js (legacy; the backend now holds the real key)
    if (typeof CONFIG !== 'undefined' && CONFIG.API_KEY && CONFIG.API_KEY.trim() !== '') {
        apiKey = CONFIG.API_KEY;
    }

    // ── Theme Management ──────────────────────────────────────────
    let currentTheme = localStorage.getItem('islamic_chatbot_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);

    function updateThemeUI() {
        if (!themeToggleIcon || !themeToggleText) return;
        if (currentTheme === 'light') {
            themeToggleIcon.className = 'fas fa-moon';
            themeToggleText.textContent = 'Dark Mode';
        } else {
            themeToggleIcon.className = 'fas fa-sun';
            themeToggleText.textContent = 'Light Mode';
        }
    }
    updateThemeUI();

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', currentTheme);
            localStorage.setItem('islamic_chatbot_theme', currentTheme);
            updateThemeUI();
        });
    }

    // Escape source text before injecting into HTML.
    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // ── Auto-scroll button ──────────────────────────────────────
    const scrollBtn = document.getElementById('scroll-bottom-btn');
    const chatContainerEl = document.getElementById('chat-container');

    chatContainerEl.addEventListener('scroll', () => {
        const threshold = 300;
        const atBottom = chatContainerEl.scrollHeight - chatContainerEl.scrollTop - chatContainerEl.clientHeight < threshold;
        scrollBtn.classList.toggle('visible', !atBottom);
    });

    scrollBtn.addEventListener('click', () => {
        chatContainerEl.scrollTo({ top: chatContainerEl.scrollHeight, behavior: 'smooth' });
    });

    // ── Hijri Date ──────────────────────────────────────────────
    function updateHijriDate() {
        const el = document.getElementById('hijri-date-text');
        if (!el) return;
        const now = new Date();
        // Simple estimation: Hijri year ≈ 2025 - 622 = 1403 (approx)
        // Use Intl API where available
        try {
            const formatter = new Intl.DateTimeFormat('en-u-ca-islamic', {
                day: 'numeric', month: 'long', year: 'numeric'
            });
            el.textContent = formatter.format(now);
        } catch {
            // Fallback: rough calculation
            const day = now.getDate();
            const month = now.getMonth() + 1;
            const year = now.getFullYear();
            const hijriYear = Math.floor((year - 622) * 0.97);
            el.textContent = `${day}/${month}/${hijriYear} AH`;
        }
    }
    updateHijriDate();

    // ── Initialization ───────────────────────────────────────────
    if (chats.length === 0) {
        createNewChat();
    } else {
        if (!currentChatId || !chats.find(c => c.id === currentChatId)) {
            currentChatId = chats[0].id;
        }
        loadChat(currentChatId);
    }
    renderHistory();


    // ════════════════════════════════════════════════════════════
    // SPEECH-TO-TEXT (STT) — Web Speech API (free, no API key)
    // Works in Chrome, Edge, Safari. Firefox needs a flag enabled.
    // ════════════════════════════════════════════════════════════

    function initSTT() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            micBtn.title = 'Voice input not supported in this browser. Use Chrome or Edge.';
            micBtn.style.opacity = '0.4';
            micBtn.style.cursor = 'not-allowed';
            return null;
        }

        const rec = new SpeechRecognition();
        rec.continuous = false;   // stop after one sentence
        rec.interimResults = true;    // show partial results while speaking
        rec.lang = 'en-US'; // default: English
        // To support Urdu: rec.lang = 'ur-PK';
        // To support Arabic: rec.lang = 'ar-SA';
        // Could auto-detect based on madhab, but English is safest default

        rec.onstart = () => {
            isRecording = true;
            micBtn.classList.add('recording');
            sttStatus.style.display = 'flex';
            sttStatusText.textContent = 'Listening... speak now';
        };

        rec.onresult = (event) => {
            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    final += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            // Show interim in input while speaking
            userInput.value = final || interim;
            if (final) {
                sttStatusText.textContent = 'Got it! Processing...';
            }
        };

        rec.onend = () => {
            isRecording = false;
            micBtn.classList.remove('recording');
            sttStatus.style.display = 'none';
            // Auto-send if we captured something
            if (userInput.value.trim()) {
                setTimeout(() => sendMessage(), 300);
            }
        };

        rec.onerror = (event) => {
            isRecording = false;
            micBtn.classList.remove('recording');
            sttStatus.style.display = 'none';
            if (event.error === 'not-allowed') {
                alert('Microphone access denied. Please allow microphone access in your browser settings.');
            } else if (event.error !== 'aborted') {
                console.warn('STT error:', event.error);
            }
        };

        return rec;
    }

    recognition = initSTT();

    micBtn.addEventListener('click', () => {
        if (!recognition) return;
        if (isRecording) {
            recognition.stop();
        } else {
            userInput.value = '';
            recognition.start();
        }
    });

    sttCancelBtn.addEventListener('click', () => {
        if (recognition && isRecording) {
            recognition.abort();
        }
    });


    // ════════════════════════════════════════════════════════════
    // TEXT-TO-SPEECH (TTS) — Web Speech Synthesis (free, built-in)
    // ════════════════════════════════════════════════════════════

    // Speak text using the backend /speak endpoint (native edge-tts voices for
    // English/Urdu/Arabic). This works on ANY machine — no OS voice install
    // needed — unlike the browser's speechSynthesis.
    async function speakText(text, onended) {
        stopSpeech();

        // Strip HTML/citations/markdown so they aren't read aloud.
        const cleanText = (text || '')
            .replace(/<[^>]*>/g, '')
            .replace(/\[.*?\]/g, '')
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .trim();
        if (!cleanText) return;

        const lang = detectSpeechLang(cleanText);   // 'ur' | 'ar' | 'en'

        try {
            const resp = await fetch(`${BACKEND_URL}/speak`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: cleanText, lang: lang })
            });
            if (!resp.ok) throw new Error('speak failed: ' + resp.status);

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            currentAudio = audio;
            audio.onended = () => {
                URL.revokeObjectURL(url);
                if (currentAudio === audio) currentAudio = null;
                if (typeof onended === 'function') onended();
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                if (currentAudio === audio) currentAudio = null;
                if (typeof onended === 'function') onended();
            };
            await audio.play();
        } catch (e) {
            console.error(e);
            if (typeof onended === 'function') onended();  // reset the button icon
        }
    }

    // Decide the spoken language from the text.
    // Urdu-specific letters -> Urdu; other Arabic-block -> Arabic; else English
    // (covers English AND Roman-Urdu, which is Latin script).
    function detectSpeechLang(text) {
        const t = text || '';
        if (/[\u067E\u0686\u0698\u06A9\u06AF\u06BA\u06BE\u06C1\u06CC\u06D2\u0679\u0688\u0691]/.test(t)) {
            return 'ur';
        }
        if (/[\u0600-\u06FF]/.test(t)) {
            return 'ar';
        }
        return 'en';
    }

    function stopSpeech() {
        if (currentAudio) {
            try { currentAudio.pause(); } catch (e) { }
            currentAudio = null;
        }
        // Also stop any legacy browser speech, just in case.
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }
    }


    // ════════════════════════════════════════════════════════════
    // WHATSAPP INTEGRATION — DISABLED
    // Uncomment this entire section when you're ready to use it.
    // You'll need WHATSAPP_TOKEN and PHONE_NUMBER_ID in config.js
    // ════════════════════════════════════════════════════════════

    /*
    const whatsappFab = document.getElementById('whatsapp-fab');

    async function sendToWhatsApp(message, recipientPhone) {
        // recipientPhone format: '923001234567' (country code + number, no +)

        const token         = CONFIG.WHATSAPP_TOKEN;       // add to config.js
        const phoneNumberId = CONFIG.PHONE_NUMBER_ID;      // add to config.js

        if (!token || !phoneNumberId) {
            console.error('WhatsApp config missing. Add WHATSAPP_TOKEN and PHONE_NUMBER_ID to config.js');
            return;
        }

        const url = `https://graph.facebook.com/v18.0/${phoneNumberId}/messages`;

        const body = {
            messaging_product: 'whatsapp',
            recipient_type: 'individual',
            to: recipientPhone,
            type: 'text',
            text: { body: message }
        };

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });
            const data = await response.json();
            console.log('WhatsApp sent:', data);
            return data;
        } catch (err) {
            console.error('WhatsApp send error:', err);
        }
    }

    // Webhook handler — set this URL in your Meta App Dashboard
    // POST /webhook receives incoming WhatsApp messages
    async function handleWhatsAppWebhook(webhookBody) {
        const entry   = webhookBody.entry?.[0];
        const changes = entry?.changes?.[0];
        const message = changes?.value?.messages?.[0];

        if (!message || message.type !== 'text') return;

        const from = message.from;           // sender's phone number
        const text = message.text.body;      // message content

        // Get madhab from a stored mapping (phone -> madhab)
        const madhab = localStorage.getItem(`wa_madhab_${from}`) || 'General';

        // Call your RAG backend here
        const answer = await callGeminiAPI(text, madhab, []);

        // Send reply back
        await sendToWhatsApp(answer, from);
    }

    // FAB click — share last bot response to WhatsApp Web (no API needed)
    // This opens wa.me with the last answer pre-filled — useful for testing
    if (whatsappFab) {
        whatsappFab.addEventListener('click', () => {
            const chat = chats.find(c => c.id === currentChatId);
            if (!chat) return;

            const lastBot = [...chat.messages].reverse().find(m => m.sender === 'bot');
            if (!lastBot) return;

            const text = encodeURIComponent(lastBot.text.substring(0, 1000));
            window.open(`https://wa.me/?text=${text}`, '_blank');
        });
    }
    */


    // ════════════════════════════════════════════════════════════
    // CHAT HISTORY MANAGEMENT
    // ════════════════════════════════════════════════════════════

    function createNewChat() {
        const newChat = {
            id: Date.now().toString(),
            title: 'New Chat',
            messages: [],
            madhab: null,
            timestamp: Date.now()
        };
        chats.unshift(newChat);
        saveChats();
        loadChat(newChat.id);
    }

    function saveChats() {
        localStorage.setItem('islamic_chatbot_chats', JSON.stringify(chats));
        renderHistory();
    }

    function loadChat(id) {
        currentChatId = id;
        localStorage.setItem('islamic_chatbot_current_id', currentChatId);

        const chat = chats.find(c => c.id === id);
        if (!chat) return;

        chatContainer.innerHTML = '';
        chatContainer.appendChild(madhabSelection);

        if (chat.madhab) {
            setMadhabUI(chat.madhab);
            madhabSelection.style.display = 'none';
        } else if (chat.messages.length === 0) {
            resetMadhabUI();
            madhabSelection.style.display = 'block';
        } else {
            // User already sent messages without selecting — default to General
            chat.madhab = 'General';
            setMadhabUI('General');
            madhabSelection.style.display = 'none';
        }

        chat.messages.forEach(msg => {
            appendMessageUI(msg.sender, msg.text, false, msg.meta);
        });

        renderHistory();
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }

    function deleteChat(id, e) {
        e.stopPropagation();
        if (confirm('Are you sure you want to delete this chat?')) {
            chats = chats.filter(c => c.id !== id);
            saveChats();
            if (chats.length === 0) {
                createNewChat();
            } else if (currentChatId === id) {
                loadChat(chats[0].id);
            } else {
                renderHistory();
            }
        }
    }

    function renameChat(id, e) {
        e.stopPropagation();
        const chat = chats.find(c => c.id === id);
        if (chat) {
            const newTitle = prompt('Enter new chat title:', chat.title);
            if (newTitle) {
                chat.title = newTitle;
                saveChats();
            }
        }
    }

    function renderHistory() {
        historyList.innerHTML = '';
        chats.forEach(chat => {
            const item = document.createElement('div');
            item.classList.add('history-item');
            if (chat.id === currentChatId) item.classList.add('active');

            const isUrduTitle = isUrduText(chat.title);
            item.innerHTML = `
                <i class="fas fa-comment-alt"></i>
                <span class="title ${isUrduTitle ? 'font-urdu-ui' : ''}" ${isUrduTitle ? 'lang="ur"' : ''}>${chat.title}</span>
                <div class="history-actions">
                    <button class="action-btn edit-btn"><i class="fas fa-edit"></i></button>
                    <button class="action-btn delete-btn"><i class="fas fa-trash"></i></button>
                </div>
            `;

            item.addEventListener('click', () => loadChat(chat.id));
            item.querySelector('.edit-btn').addEventListener('click', (e) => renameChat(chat.id, e));
            item.querySelector('.delete-btn').addEventListener('click', (e) => deleteChat(chat.id, e));

            historyList.appendChild(item);
        });
    }

    newChatBtn.addEventListener('click', createNewChat);


    // ════════════════════════════════════════════════════════════
    // MADHAB SELECTION
    // ════════════════════════════════════════════════════════════

    madhabCards.forEach(card => {
        card.addEventListener('click', () => {
            const selected = card.getAttribute('data-madhab');
            const chat = chats.find(c => c.id === currentChatId);
            if (!chat) return;

            const prevMadhab = chat.madhab;
            chat.madhab = selected;
            setMadhabUI(selected);
            madhabSelection.style.display = 'none';
            saveChats();

            // Only push a bot message if this is the very first selection
            if (!prevMadhab) {
                const msg = `You have selected the **${selected}** approach. How can I assist you?`;
                chat.messages.push({ sender: 'bot', text: msg });
                saveChats();
                appendMessageUI('bot', msg, false);
            }
        });
    });

    function setMadhabUI(madhab) {
        sidebarMadhabName.textContent = madhab + (madhab === 'General' ? '' : ' Madhab');
    }

    function resetMadhabUI() {
        sidebarMadhabName.textContent = 'General';
    }

    changeMadhabBtn.addEventListener('click', () => {
        const chat = chats.find(c => c.id === currentChatId);
        if (chat) {
            chat.madhab = null;
            saveChats();
        }
        madhabSelection.style.display = 'block';
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    });


    // ════════════════════════════════════════════════════════════
    // SEND MESSAGE & API CALL
    // ════════════════════════════════════════════════════════════

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Stop any ongoing speech when user sends a new message
        stopSpeech();

        const chat = chats.find(c => c.id === currentChatId);
        if (!chat) return;

        // If user bypasses madhab card selection, auto-dismiss cards and default to General
        if (!chat.madhab) {
            chat.madhab = 'General';
            setMadhabUI('General');
            madhabSelection.style.display = 'none';
        }

        // The most recent bot answer — sent along so a follow-up like
        // "explain in urdu" can re-express it instead of starting a new topic.
        const lastBot = [...chat.messages].reverse().find(m => m.sender === 'bot');
        const prevAnswer = lastBot ? lastBot.text : '';

        chat.messages.push({ sender: 'user', text: text });

        // Move this chat to the top of the recent list
        chats = chats.filter(c => c.id !== chat.id);
        chats.unshift(chat);

        saveChats();
        appendMessageUI('user', text, true);
        userInput.value = '';

        // Loading indicator — mosque pulse
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot-message', 'loading-msg');
        loadingDiv.innerHTML = `<div class="message-content"><div class="thinking-icon"><i class="fas fa-mosque"></i></div><span>Reflecting<span class="typing-dots"><span></span><span></span><span></span></span></span></div>`;
        chatContainer.insertBefore(loadingDiv, madhabSelection);
        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });

        try {
            const result = await askBackend(text, chat.madhab || 'General', prevAnswer);
            loadingDiv.remove();

            if (!result || result.ok === false) {
                const errMsg = (result && result.error) ? result.error : 'something went wrong.';
                appendMessageUI('bot', 'Sorry, ' + errMsg, false);
                return;
            }

            const answer = result.answer || '';
            const meta = {
                mode: result.mode,
                confidence_percent: result.confidence_percent,
                confidence_label: result.confidence_label,
                confidence_reason: result.confidence_reason,
                consensus: result.consensus,
                sources: result.sources,
                out_of_scope: result.out_of_scope
            };

            chat.messages.push({ sender: 'bot', text: answer, meta: meta });
            saveChats();
            appendMessageUI('bot', answer, true, meta);

            // Auto-rename chat from first question
            if (chat.title === 'New Chat' && chat.messages.length <= 3) {
                chat.title = text.substring(0, 25) + (text.length > 25 ? '...' : '');
                saveChats();
            }

        } catch (error) {
            loadingDiv.remove();
            appendMessageUI('bot',
                'Sorry, I could not reach the server. Make sure the backend is running ' +
                '(python app.py) and reachable at ' + BACKEND_URL + '.', false);
            console.error(error);
        }
    }


    // ════════════════════════════════════════════════════════════
    // RENDER MESSAGES (with TTS speak button on bot messages)
    // ════════════════════════════════════════════════════════════

    function buildSourcesAccordion(sources) {
        const container = document.createElement('div');
        container.classList.add('sources-accordion');

        const topics = (sources.fiqh || []).map(f => f.topic_name || f.topic_id).filter(Boolean);
        if (topics.length) {
            const line = document.createElement('div');
            line.className = 'src-static-line';
            line.innerHTML = `<strong>Topic:</strong> ${escapeHtml(topics.join(', '))}`;
            const hasConsensus = (sources.fiqh || []).some(f => f.consensus === false);
            if (hasConsensus) {
                line.innerHTML += ` <span class="src-differ">(Scholars differ)</span>`;
            }
            container.appendChild(line);
        }

        (sources.quran || []).forEach(q => {
            const ref = q.ref || '';
            const surah = q.surah ? ` - ${q.surah}` : '';
            const wrapper = buildClickableLine(`Qur'an: (${ref}${surah})`, 'quran', q);
            container.appendChild(wrapper);
        });

        (sources.hadith || []).forEach(h => {
            const hasText = (h.text_ar && h.text_ar.trim()) || (h.text_en && h.text_en.trim());
            if (!hasText) return;
            const ref = (h.ref || '').replace(/&#39;|&apos;/g, "'");
            const chapter = h.chapter ? ` - ${h.chapter}` : '';
            const wrapper = buildClickableLine(`Hadith: (${ref}${chapter})`, 'hadith', h);
            container.appendChild(wrapper);
        });

        return container;
    }

    function buildClickableLine(label, type, data) {
        const wrapper = document.createElement('div');
        wrapper.className = 'src-clickable-row';

        const header = document.createElement('div');
        header.className = 'src-row-header';

        const text = document.createElement('span');
        text.className = 'src-row-text';
        text.textContent = label;

        const chevron = document.createElement('i');
        chevron.className = 'fas fa-chevron-down src-row-chevron';

        header.appendChild(text);
        header.appendChild(chevron);

        const panel = document.createElement('div');
        panel.className = 'src-row-panel';
        panel.style.display = 'none';

        header.addEventListener('click', () => {
            const isOpen = panel.style.display !== 'none';
            panel.style.display = isOpen ? 'none' : 'block';
            header.classList.toggle('open', !isOpen);
            if (!isOpen && !panel.hasChildNodes()) {
                panel.appendChild(buildTrilingualContent(data, type));
            }
        });

        wrapper.appendChild(header);
        wrapper.appendChild(panel);
        return wrapper;
    }


    function buildTrilingualContent(item, type) {
        const div = document.createElement('div');
        div.className = 'source-content';

        if (type === 'quran') {
            if (item.arabic) {
                const block = document.createElement('div');
                block.className = 'src-lang-block';
                block.dir = 'rtl';
                block.innerHTML = `<span class="src-lang-label">ARABIC</span><p class="src-lang-text src-lang-arabic">${escapeHtml(item.arabic)}</p>`;
                div.appendChild(block);
            }
            if (item.english) {
                const block = document.createElement('div');
                block.className = 'src-lang-block';
                block.dir = 'ltr';
                block.innerHTML = `<span class="src-lang-label">ENGLISH</span><p class="src-lang-text src-lang-english">${escapeHtml(item.english)}</p>`;
                div.appendChild(block);
            }
            if (item.urdu) {
                const block = document.createElement('div');
                block.className = 'src-lang-block';
                block.dir = 'rtl';
                block.innerHTML = `<span class="src-lang-label">URDU</span><p class="src-lang-text src-lang-urdu">${escapeHtml(item.urdu)}</p>`;
                div.appendChild(block);
            }
        } else if (type === 'hadith') {
            if (item.text_ar && item.text_ar.trim()) {
                const block = document.createElement('div');
                block.className = 'src-lang-block';
                block.dir = 'rtl';
                block.innerHTML = `<span class="src-lang-label">ARABIC</span><p class="src-lang-text src-lang-arabic">${escapeHtml(item.text_ar)}</p>`;
                div.appendChild(block);
            }
            if (item.text_en && item.text_en.trim()) {
                const block = document.createElement('div');
                block.className = 'src-lang-block';
                block.dir = 'ltr';
                block.innerHTML = `<span class="src-lang-label">ENGLISH</span><p class="src-lang-text src-lang-english">${escapeHtml(item.text_en)}</p>`;
                div.appendChild(block);
            }
        }

        return div;
    }

    function buildMetaBadges(meta) {
        if (!meta || meta.mode === 'greeting' || meta.out_of_scope) return '';

        const pct = meta.confidence_percent;
        const label = meta.confidence_label || '';
        const lvl = label.toLowerCase().replace(/\s+/g, '-');
        const hasConsensus = meta.consensus === true || meta.consensus === false;

        let html = '<div class="msg-meta">';

        if (pct != null) {
            html += `<span class="conf-badge conf-${lvl}">Confidence: ${pct}% · ${escapeHtml(label)}</span>`;
        }

        if (hasConsensus) {
            html += meta.consensus
                ? `<span class="cons-badge cons-yes"><i class="fas fa-check-circle"></i> All madhhabs agree</span>`
                : `<span class="cons-badge cons-no"><i class="fas fa-info-circle"></i> Scholars differ</span>`;
        }

        html += '</div>';
        return html;
    }

    function appendMessageUI(sender, text, animate, meta) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');

        if (isUrduText(text)) {
            contentDiv.classList.add('font-urdu');
            contentDiv.setAttribute('lang', 'ur');
        }

        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Make any URL clickable (opens in a new tab). Covers the IslamQA
        // referral link shown on low-confidence answers.
        formattedText = formattedText.replace(
            /(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" rel="noopener">$1</a>'
        );
        formattedText = formattedText.replace(/\n/g, '<br>');

        // Show the Sources button only for real answers that actually carry
        // citations. Referral / greeting / capability / low-confidence replies
        // ("ask a scholar", IslamQA link, etc.) have no sources, so hide it.
        const noSourceModes = ['greeting', 'capability', 'low_confidence', 'reexpress'];
        const s = (meta && meta.sources) || {};
        const realSourceCount =
            (s.fiqh   || []).filter(Boolean).length +
            (s.quran  || []).filter(Boolean).length +
            (s.hadith || []).filter(Boolean).length;
        const hasSources = sender === 'bot'
            && !noSourceModes.includes(meta && meta.mode)
            && realSourceCount > 0;
        const metaBadges = buildMetaBadges(meta);
        const speakButton = sender === 'bot'
            ? `<button class="speak-btn" title="Read aloud"><i class="fas fa-volume-up"></i></button>`
            : '';
        const copyButton = sender === 'bot'
            ? `<button class="copy-btn" title="Copy message"><i class="fas fa-copy"></i></button>`
            : '';
        const sourcesButton = hasSources
            ? `<button class="sources-btn" title="View sources"><i class="fas fa-book-open"></i> Sources</button>`
            : '';

        contentDiv.innerHTML = `<p>${formattedText}</p>${metaBadges}${speakButton}${copyButton}${sourcesButton}`;

        if (hasSources) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.classList.add('sources-dropdown');
            sourcesDiv.style.display = 'none';
            contentDiv.appendChild(sourcesDiv);

            const srcBtn = contentDiv.querySelector('.sources-btn');
            let accordionBuilt = false;
            srcBtn.addEventListener('click', () => {
                const isOpen = sourcesDiv.style.display !== 'none';
                sourcesDiv.style.display = isOpen ? 'none' : 'block';
                srcBtn.classList.toggle('active', !isOpen);
                if (!isOpen && !accordionBuilt) {
                    const accordion = buildSourcesAccordion(meta.sources);
                    sourcesDiv.appendChild(accordion);
                    accordionBuilt = true;
                }
            });
        }

        if (sender === 'bot') {
            const speakBtn = contentDiv.querySelector('.speak-btn');
            let isSpeaking = false;

            const resetIcon = () => {
                speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
                isSpeaking = false;
            };

            speakBtn.addEventListener('click', () => {
                if (isSpeaking) {
                    stopSpeech();
                    resetIcon();
                } else {
                    stopSpeech();
                    speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                    isSpeaking = true;
                    speakText(text, resetIcon).then(() => {
                        if (isSpeaking) speakBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    });
                }
            });

            const copyBtn = contentDiv.querySelector('.copy-btn');
            copyBtn.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(text);
                    copyBtn.classList.add('copied');
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 2000);
                } catch {
                    copyBtn.innerHTML = '<i class="fas fa-times"></i> Failed';
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                    }, 1500);
                }
            });
        }

        msgDiv.appendChild(contentDiv);

        if (madhabSelection.parentNode === chatContainer) {
            chatContainer.insertBefore(msgDiv, madhabSelection);
        } else {
            chatContainer.appendChild(msgDiv);
        }

        requestAnimationFrame(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }


    // ════════════════════════════════════════════════════════════
    // BACKEND CALL  (RAG pipeline + grounded Gemini, via Flask /ask)
    // ════════════════════════════════════════════════════════════

    async function askBackend(question, madhab, prevAnswer) {
        const response = await fetch(`${BACKEND_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                madhab: madhab || 'General',
                prev_answer: prevAnswer || ''
            })
        });
        if (!response.ok) {
            let detail = response.status;
            try { const j = await response.json(); detail = j.error || detail; } catch (e) { }
            throw new Error('Backend error: ' + detail);
        }
        return await response.json();
    }



    // ════════════════════════════════════════════════════════════
    // GEMINI API (legacy direct call — kept only for the commented-out
    // client-side WhatsApp block; the main chat now uses askBackend above)
    // ════════════════════════════════════════════════════════════

    async function callGeminiAPI(prompt, madhab, chatHistory) {
        const madhabContext = madhab
            ? `You are a knowledgeable Islamic scholar following the ${madhab} school of thought. Provide guidance based specifically on this school's rulings and methodology. If there are differences of opinion, mention them. Keep answers concise and direct.`
            : `You are a knowledgeable Islamic scholar. Provide general Islamic guidance based on the Quran and Sunnah. Keep answers concise and direct.`;

        const apiContents = [];
        let contextAdded = false;

        chatHistory.forEach(msg => {
            const role = msg.sender === 'user' ? 'user' : 'model';
            let text = msg.text;

            if (role === 'user' && !contextAdded) {
                text = `${madhabContext}\n\n${text}`;
                contextAdded = true;
            }

            apiContents.push({ role, parts: [{ text }] });
        });

        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-09-2025:generateContent?key=${apiKey}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contents: apiContents })
        });

        if (!response.ok) throw new Error('API Request Failed');

        const data = await response.json();
        return data.candidates[0].content.parts[0].text;
    }


    // ── Event Listeners ──────────────────────────────────────────
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    userInput.addEventListener('input', () => {
        if (isUrduText(userInput.value)) {
            userInput.classList.add('font-urdu-ui');
            userInput.setAttribute('lang', 'ur');
        } else {
            userInput.classList.remove('font-urdu-ui');
            userInput.removeAttribute('lang');
        }
    });

    // ── Show enter button after minimum 2.8s ────────────────────
    showEnterBtn();
});
