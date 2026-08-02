/**
 * resize.js — Panel Resize Logic for Islamic Fiqhbot
 *
 * Makes the sidebar resizable by dragging its right edge.
 * Persists the chosen width to localStorage so it survives page reloads.
 */

(function () {
    const STORAGE_KEY = 'fiqhbot_sidebar_width';
    const SIDEBAR_MIN = 200;   // px
    const SIDEBAR_MAX = 500;   // px
    const SIDEBAR_DEFAULT = 300; // px

    const sidebar = document.getElementById('sidebar');
    const handle  = document.getElementById('sidebar-resize-handle');

    if (!sidebar || !handle) return;

    // ── Restore persisted width ──────────────────────────────────
    const saved = parseInt(localStorage.getItem(STORAGE_KEY), 10);
    if (saved && saved >= SIDEBAR_MIN && saved <= SIDEBAR_MAX) {
        sidebar.style.width = saved + 'px';
    }

    // ── Drag State ───────────────────────────────────────────────
    let isResizing  = false;
    let startX      = 0;
    let startWidth  = 0;

    // ── Mousedown on handle ──────────────────────────────────────
    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isResizing = true;
        startX     = e.clientX;
        startWidth = sidebar.getBoundingClientRect().width;

        handle.classList.add('is-resizing');
        sidebar.classList.add('is-resizing');

        // Prevent text selection anywhere while dragging
        document.body.style.userSelect    = 'none';
        document.body.style.cursor        = 'col-resize';
        document.documentElement.style.cursor = 'col-resize';
    });

    // ── Mousemove anywhere on document ───────────────────────────
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const delta    = e.clientX - startX;
        const newWidth = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startWidth + delta));

        sidebar.style.width = newWidth + 'px';
    });

    // ── Mouseup — end resize ─────────────────────────────────────
    document.addEventListener('mouseup', () => {
        if (!isResizing) return;
        isResizing = false;

        handle.classList.remove('is-resizing');
        sidebar.classList.remove('is-resizing');

        document.body.style.userSelect    = '';
        document.body.style.cursor        = '';
        document.documentElement.style.cursor = '';

        // Persist width
        const w = parseInt(sidebar.style.width, 10);
        if (!isNaN(w)) localStorage.setItem(STORAGE_KEY, w);
    });

    // ── Double-click handle — reset to default width ─────────────
    handle.addEventListener('dblclick', () => {
        sidebar.style.width = SIDEBAR_DEFAULT + 'px';
        localStorage.setItem(STORAGE_KEY, SIDEBAR_DEFAULT);
    });

    // ── Touch support ─────────────────────────────────────────────
    handle.addEventListener('touchstart', (e) => {
        const touch = e.touches[0];
        isResizing = true;
        startX     = touch.clientX;
        startWidth = sidebar.getBoundingClientRect().width;
        handle.classList.add('is-resizing');
        sidebar.classList.add('is-resizing');
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchmove', (e) => {
        if (!isResizing) return;
        const touch    = e.touches[0];
        const delta    = touch.clientX - startX;
        const newWidth = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, startWidth + delta));
        sidebar.style.width = newWidth + 'px';
    }, { passive: true });

    document.addEventListener('touchend', () => {
        if (!isResizing) return;
        isResizing = false;
        handle.classList.remove('is-resizing');
        sidebar.classList.remove('is-resizing');
        const w = parseInt(sidebar.style.width, 10);
        if (!isNaN(w)) localStorage.setItem(STORAGE_KEY, w);
    });

    // ── Collapse / Expand Toggle ──────────────────────────────────
    const toggleBtn = document.getElementById('sidebar-toggle');
    const reopenBtn = document.getElementById('sidebar-reopen-btn');

    if (toggleBtn) {
        const COLLAPSE_KEY = 'fiqhbot_sidebar_collapsed';
        const panelCloseSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/><path d="m14 9-3 3 3 3"/></svg>';
        const panelOpenSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/><path d="m11 9 3 3-3 3"/></svg>';

        function setToggleIcon(collapsed) {
            toggleBtn.innerHTML = collapsed ? panelOpenSvg : panelCloseSvg;
            toggleBtn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
        }

        function expandSidebar() {
            sidebar.classList.remove('collapsed');
            localStorage.setItem(COLLAPSE_KEY, 'false');
            setToggleIcon(false);
        }

        function collapseSidebar() {
            sidebar.classList.add('collapsed');
            localStorage.setItem(COLLAPSE_KEY, 'true');
            setToggleIcon(true);
        }

        // Restore collapsed state
        if (localStorage.getItem(COLLAPSE_KEY) === 'true') {
            sidebar.classList.add('collapsed');
            setToggleIcon(true);
        }

        toggleBtn.addEventListener('click', () => {
            if (sidebar.classList.contains('collapsed')) {
                expandSidebar();
            } else {
                collapseSidebar();
            }
        });

        // Mosque logo in the rail also expands when clicked
        if (reopenBtn) {
            reopenBtn.addEventListener('click', (e) => {
                if (sidebar.classList.contains('collapsed')) {
                    e.stopPropagation();
                    expandSidebar();
                }
            });
        }
    }
})();
