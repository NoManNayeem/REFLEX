/**
 * Enhanced security and error handling utilities for REFLEX frontend
 * Add these functions to app.js
 */

// ==================== SECURITY UTILITIES ====================

/**
 * Sanitize HTML content to prevent XSS attacks
 * Uses DOMPurify for comprehensive sanitization
 */
function sanitizeMessage(html) {
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(html, {
            ALLOWED_TAGS: [
                'p', 'br', 'strong', 'em', 'code', 'pre', 'a',
                'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td'
            ],
            ALLOWED_ATTR: ['href', 'target', 'rel', 'class'],
            ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i
        });
    }
    // Fallback if DOMPurify not loaded
    console.warn('DOMPurify not loaded, using basic escaping');
    return html.replace(/[&<>"']/g, (char) => {
        const escapeMap = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        };
        return escapeMap[char];
    });
}

// ==================== NETWORK UTILITIES ====================

/**
 * Fetch with retry logic and exponential backoff
 * Handles rate limiting and transient network errors
 */
async function fetchWithRetry(url, options = {}, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const response = await fetch(url, options);

            // Handle rate limiting (429 Too Many Requests)
            if (response.status === 429) {
                const retryAfter = response.headers.get('Retry-After');
                const delay = retryAfter ? parseInt(retryAfter) * 1000 : Math.pow(2, attempt) * 1000;

                console.warn(`Rate limited. Retrying after ${delay}ms...`);
                showWarning(`Too many requests. Waiting ${delay / 1000}s before retry...`);

                await new Promise(r => setTimeout(r, delay));
                continue;
            }

            // Success or non-retryable error
            if (response.ok || attempt === maxRetries - 1) {
                return response;
            }

            // Retryable errors (5xx server errors)
            if (response.status >= 500) {
                const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
                console.warn(`Server error ${response.status}. Retrying in ${delay}ms...`);
                await new Promise(r => setTimeout(r, delay));
                continue;
            }

            // Non-retryable client error
            return response;

        } catch (error) {
            // Network error or timeout
            if (attempt === maxRetries - 1) {
                throw error;
            }

            const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
            console.warn(`Network error: ${error.message}. Retrying in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
        }
    }
}

// ==================== ERROR DISPLAY ====================

/**
 * Show error banner with auto-dismiss
 */
function showError(message, details = null, duration = 5000) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-banner';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <div class="error-text">
                <strong>Error</strong>
                <p>${escapeHtml(message)}</p>
                ${details ? `<small>${escapeHtml(details)}</small>` : ''}
            </div>
            <button class="close-error" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(errorDiv);

    if (duration > 0) {
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, duration);
    }
}

/**
 * Show warning banner
 */
function showWarning(message, duration = 3000) {
    const warningDiv = document.createElement('div');
    warningDiv.className = 'warning-banner';
    warningDiv.innerHTML = `
        <div class="warning-content">
            <i class="fas fa-exclamation-circle"></i>
            <div class="warning-text">
                <strong>Warning</strong>
                <p>${escapeHtml(message)}</p>
            </div>
            <button class="close-warning" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(warningDiv);

    if (duration > 0) {
        setTimeout(() => {
            if (warningDiv.parentElement) {
                warningDiv.remove();
            }
        }, duration);
    }
}

/**
 * Show success banner
 */
function showSuccess(message, duration = 3000) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-banner';
    successDiv.innerHTML = `
        <div class="success-content">
            <i class="fas fa-check-circle"></i>
            <div class="success-text">
                <p>${escapeHtml(message)}</p>
            </div>
            <button class="close-success" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(successDiv);

    if (duration > 0) {
        setTimeout(() => {
            if (successDiv.parentElement) {
                successDiv.remove();
            }
        }, duration);
    }
}

// ==================== INPUT VALIDATION ====================

/**
 * Validate message input before sending
 */
function validateMessage(message) {
    if (!message || !message.trim()) {
        showError('Message cannot be empty');
        return false;
    }

    if (message.length > 10000) {
        showError('Message is too long', 'Maximum length is 10,000 characters');
        return false;
    }

    return true;
}

/**
 * Validate URL format
 */
function validateUrl(url) {
    try {
        const urlObj = new URL(url);
        return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
    } catch (e) {
        return false;
    }
}

// ==================== USAGE EXAMPLE ====================

/**
 * Example: Update the sendMessage function to use these utilities
 */
async function sendMessageSecure() {
    const message = elements.messageInput.value.trim();

    // Validate input
    if (!validateMessage(message)) {
        return;
    }

    // Disable input
    elements.sendBtn.disabled = true;
    elements.messageInput.disabled = true;

    try {
        // Use fetchWithRetry instead of fetch
        const response = await fetchWithRetry(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: state.sessionId,
                user_id: state.userId
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
        }

        // Process streaming response...

    } catch (error) {
        console.error('Error:', error);
        showError('Failed to send message', error.message);
    } finally {
        elements.sendBtn.disabled = false;
        elements.messageInput.disabled = false;
        elements.messageInput.focus();
    }
}

/**
 * Example: Update formatMessage to use sanitization
 */
function formatMessageSecure(content) {
    if (!content) return '';

    const text = String(content);

    if (typeof marked !== 'undefined') {
        try {
            const markedOptions = {
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            };

            const html = marked.parse(text, markedOptions);
            // IMPORTANT: Sanitize the HTML output
            return sanitizeMessage(html);
        } catch (error) {
            console.warn('Markdown parsing error:', error);
            return sanitizeMessage(text);
        }
    } else {
        return sanitizeMessage(text);
    }
}

// Export functions for use in main app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        sanitizeMessage,
        fetchWithRetry,
        showError,
        showWarning,
        showSuccess,
        validateMessage,
        validateUrl
    };
}
