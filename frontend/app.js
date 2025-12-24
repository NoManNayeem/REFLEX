// REFLEX - Research Engine with Feedback-Driven Learning - Frontend Application

// API_BASE - Use localhost for browser access (works in both Docker and local dev)
// In Docker, the browser accesses localhost:8000 which is mapped to the backend container
// Note: Browser runs on host machine, so it must use localhost, not Docker service names
const API_BASE = 'http://localhost:8000/api';
console.log('API Base URL:', API_BASE);

// State management
const state = {
    sessionId: localStorage.getItem('currentSessionId') || `session_${Date.now()}`,
    userId: 'demo_user',
    messageCount: 0,
    lastTrajectory: null,
    currentTheme: localStorage.getItem('theme') || 'light',
    conversations: [],
    currentConversation: null
};

// Save session ID to localStorage
if (!localStorage.getItem('currentSessionId')) {
    localStorage.setItem('currentSessionId', state.sessionId);
}

// DOM Elements
const elements = {
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    chatMessages: document.getElementById('chatMessages'),
    themeToggle: document.getElementById('themeToggle'),
    refreshBtn: document.getElementById('refreshBtn'),
    viewSkillsBtn: document.getElementById('viewSkillsBtn'),
    trainBtn: document.getElementById('trainBtn'),
    
    // Stats
    totalTasks: document.getElementById('totalTasks'),
    successRate: document.getElementById('successRate'),
    avgReward: document.getElementById('avgReward'),
    skillCount: document.getElementById('skillCount'),
    topSkillsList: document.getElementById('topSkillsList'),
    
    // Feedback
    taskSuccess: document.getElementById('taskSuccess'),
    taskSuccessValue: document.getElementById('taskSuccessValue'),
    quality: document.getElementById('quality'),
    qualityValue: document.getElementById('qualityValue'),
    efficiency: document.getElementById('efficiency'),
    efficiencyValue: document.getElementById('efficiencyValue'),
    userFeedback: document.getElementById('userFeedback'),
    userFeedbackValue: document.getElementById('userFeedbackValue'),
    createSkill: document.getElementById('createSkill'),
    skillForm: document.getElementById('skillForm'),
    submitFeedback: document.getElementById('submitFeedback'),
    feedbackStatus: document.getElementById('feedbackStatus'),
    
    // Trajectory
    toolsUsed: document.getElementById('toolsUsed'),
    skillsApplied: document.getElementById('skillsApplied'),
    sessionId: document.getElementById('sessionId'),
    messageCountDisplay: document.getElementById('messageCount'),
    
    // Modal
    skillsModal: document.getElementById('skillsModal'),
    closeModal: document.getElementById('closeModal'),
    allSkillsList: document.getElementById('allSkillsList'),
    skillSearch: document.getElementById('skillSearch'),
    
    // Help panel
    helpBtn: document.getElementById('helpBtn'),
    helpPanel: document.getElementById('helpPanel'),
    closeHelpBtn: document.getElementById('closeHelpBtn'),
    
    // Conversations
    newChatBtn: document.getElementById('newChatBtn'),
    conversationsList: document.getElementById('conversationsList'),
    
    // Knowledge Base
    addKnowledgeBtn: document.getElementById('addKnowledgeBtn'),
    knowledgeUrl: document.getElementById('knowledgeUrl'),
    knowledgeSourcesList: document.getElementById('knowledgeSourcesList'),
    knowledgeStatus: document.getElementById('knowledgeStatus'),
    reloadKnowledgeBtn: document.getElementById('reloadKnowledgeBtn'),
    clearKnowledgeBtn: document.getElementById('clearKnowledgeBtn')
};

// Initialize application
function init() {
    console.log('🚀 Initializing REFLEX UI...');
    
    // Set theme
    document.body.setAttribute('data-theme', state.currentTheme);
    updateThemeIcon();
    
    // Event listeners
    elements.sendBtn.addEventListener('click', sendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    elements.messageInput.addEventListener('input', () => {
        elements.messageInput.style.height = 'auto';
        elements.messageInput.style.height = elements.messageInput.scrollHeight + 'px';
    });
    
    // Theme toggle
    elements.themeToggle.addEventListener('click', toggleTheme);
    
    // Refresh stats
    elements.refreshBtn.addEventListener('click', updateStats);
    
    // View skills modal
    elements.viewSkillsBtn.addEventListener('click', () => {
        openSkillsModal();
    });
    
    elements.closeModal.addEventListener('click', closeSkillsModal);
    
    // Click outside modal to close
    elements.skillsModal.addEventListener('click', (e) => {
        if (e.target === elements.skillsModal) {
            closeSkillsModal();
        }
    });
    
    // Train button
    elements.trainBtn.addEventListener('click', triggerTraining);
    
    // Feedback sliders
    elements.taskSuccess.addEventListener('input', (e) => {
        elements.taskSuccessValue.textContent = e.target.value;
    });
    
    elements.quality.addEventListener('input', (e) => {
        elements.qualityValue.textContent = e.target.value;
    });
    
    elements.efficiency.addEventListener('input', (e) => {
        elements.efficiencyValue.textContent = e.target.value;
    });
    
    elements.userFeedback.addEventListener('input', (e) => {
        elements.userFeedbackValue.textContent = e.target.value;
    });
    
    // Create skill checkbox
    elements.createSkill.addEventListener('change', (e) => {
        elements.skillForm.style.display = e.target.checked ? 'flex' : 'none';
    });
    
    // Submit feedback
    elements.submitFeedback.addEventListener('click', submitFeedback);
    
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
    
    // Skill search
    elements.skillSearch.addEventListener('input', (e) => {
        filterSkills(e.target.value);
    });
    
    // Update session info
    elements.sessionId.textContent = state.sessionId;
    
    // Load initial stats
    updateStats();
    
    // Load conversations and chat history
    loadConversations();
    loadChatHistory();
    
    // Load knowledge base
    loadKnowledgeBase();
    
    // New chat button
    if (elements.newChatBtn) {
        elements.newChatBtn.addEventListener('click', createNewConversation);
    }
    
    // Knowledge base handlers
    if (elements.addKnowledgeBtn) {
        elements.addKnowledgeBtn.addEventListener('click', addKnowledgeSource);
    }
    if (elements.reloadKnowledgeBtn) {
        elements.reloadKnowledgeBtn.addEventListener('click', reloadKnowledgeBase);
    }
    if (elements.clearKnowledgeBtn) {
        elements.clearKnowledgeBtn.addEventListener('click', clearKnowledgeBase);
    }
    
    // Load knowledge base on init
    loadKnowledgeBase();
    
    // Help panel toggle
    if (elements.helpBtn) {
        elements.helpBtn.addEventListener('click', () => {
            elements.helpPanel.classList.toggle('active');
        });
    }
    if (elements.closeHelpBtn) {
        elements.closeHelpBtn.addEventListener('click', () => {
            elements.helpPanel.classList.remove('active');
        });
    }
    
    console.log('✅ Application initialized');
}

// Load all conversations
async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE}/sessions?limit=50`);
        if (!response.ok) {
            // If endpoint doesn't exist yet, create a default conversation
            elements.conversationsList.innerHTML = '';
            renderConversationItem({
                session_id: state.sessionId,
                last_activity: new Date().toISOString(),
                message_count: 0
            }, true);
            return;
        }
        
        const data = await response.json();
        state.conversations = data.sessions || [];
        
        // Render conversations
        elements.conversationsList.innerHTML = '';
        
        if (state.conversations.length === 0) {
            // Create default conversation if none exist
            renderConversationItem({
                session_id: state.sessionId,
                last_activity: new Date().toISOString(),
                message_count: 0
            }, true);
        } else {
            // Sort by last activity (most recent first)
            state.conversations.sort((a, b) => 
                new Date(b.last_activity) - new Date(a.last_activity)
            );
            
            // Check if current session exists in list
            const currentSessionExists = state.conversations.some(c => c.session_id === state.sessionId);
            
            // If current session not in list, add it
            if (!currentSessionExists) {
                state.conversations.unshift({
                    session_id: state.sessionId,
                    last_activity: new Date().toISOString(),
                    message_count: 0
                });
            }
            
            state.conversations.forEach((conv) => {
                const isActive = conv.session_id === state.sessionId;
                renderConversationItem(conv, isActive);
            });
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        // Fallback: show current session
        elements.conversationsList.innerHTML = '';
        renderConversationItem({
            session_id: state.sessionId,
            last_activity: new Date().toISOString(),
            message_count: 0
        }, true);
    }
}

// Render a conversation item
function renderConversationItem(conversation, isActive = false, appendToList = true) {
    const item = document.createElement('div');
    item.className = `conversation-item ${isActive ? 'active' : ''}`;
    item.dataset.sessionId = conversation.session_id;
    
    // Get first message as title (or use default)
    const title = conversation.title || `Chat ${conversation.session_id.slice(-6)}`;
    const date = new Date(conversation.last_activity);
    const timeAgo = getTimeAgo(date);
    
    item.innerHTML = `
        <div class="conversation-content">
            <div class="conversation-title">${escapeHtml(title)}</div>
            <div class="conversation-meta">
                <span>${conversation.message_count || 0} messages</span>
                <span>•</span>
                <span>${timeAgo}</span>
            </div>
        </div>
        <div class="conversation-actions">
            <button class="conversation-action-btn" onclick="event.stopPropagation(); deleteConversation('${conversation.session_id}')" title="Delete">
                <i class="fas fa-trash"></i>
            </button>
        </div>
    `;
    
    item.addEventListener('click', () => {
        switchConversation(conversation.session_id);
    });
    
    if (appendToList && elements.conversationsList) {
        elements.conversationsList.appendChild(item);
    }
    
    return item;
}

// Get time ago string
function getTimeAgo(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Switch to a different conversation
async function switchConversation(sessionId) {
    if (sessionId === state.sessionId) return;
    
    // Update state
    state.sessionId = sessionId;
    localStorage.setItem('currentSessionId', sessionId);
    
    // Update active conversation in UI
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.dataset.sessionId === sessionId);
    });
    
    // Clear current chat
    elements.chatMessages.innerHTML = '';
    
    // Load new conversation history
    await loadChatHistory();
    
    // Update session info
    if (elements.sessionId) {
        elements.sessionId.textContent = sessionId;
    }
    
    // Reset message count
    state.messageCount = 0;
    if (elements.messageCountDisplay) {
        elements.messageCountDisplay.textContent = '0';
    }
}

// Create new conversation
function createNewConversation() {
    const newSessionId = `session_${Date.now()}`;
    switchConversation(newSessionId);
    
    // Add to conversations list
    const newConv = {
        session_id: newSessionId,
        last_activity: new Date().toISOString(),
        message_count: 0
    };
    
    state.conversations.unshift(newConv);
    
    // Remove active class from all items
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add new conversation at the top (don't append automatically)
    const newItem = renderConversationItem(newConv, true, false);
    if (elements.conversationsList && newItem) {
        if (elements.conversationsList.firstChild) {
            elements.conversationsList.insertBefore(newItem, elements.conversationsList.firstChild);
        } else {
            elements.conversationsList.appendChild(newItem);
        }
    }
}

// Create conversation element (helper function)
function createConversationElement(conversation, isActive = false) {
    return renderConversationItem(conversation, isActive);
}

// Delete conversation
async function deleteConversation(sessionId) {
    if (!confirm('Are you sure you want to delete this conversation?')) return;
    
    // Remove from UI
    const item = document.querySelector(`[data-session-id="${sessionId}"]`);
    if (item) item.remove();
    
    // If it's the current session, create a new one
    if (sessionId === state.sessionId) {
        createNewConversation();
    }
    
    // Remove from state
    state.conversations = state.conversations.filter(c => c.session_id !== sessionId);
}

// Make deleteConversation available globally
window.deleteConversation = deleteConversation;

// Load chat history from database
async function loadChatHistory() {
    try {
        // Clear current messages first
        const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
        
        const response = await fetch(`${API_BASE}/chat/history?session_id=${state.sessionId}`);
        if (!response.ok) {
            // If no history, show welcome message
            if (!welcomeMsg && elements.chatMessages.children.length === 0) {
                showWelcomeMessage();
            }
            return;
        }
        
        const data = await response.json();
        if (data.messages && data.messages.length > 0) {
            // Remove welcome message if history exists
            if (welcomeMsg) welcomeMsg.remove();
            
            // Clear any existing messages
            elements.chatMessages.innerHTML = '';
            
            // Load messages
            data.messages.forEach(msg => {
                addMessage(msg.role, msg.content, {
                    tools: msg.tools_used || [],
                    skills: msg.skills_applied || []
                }, false); // Don't scroll for each historical message
            });
            
            // Scroll to bottom after loading all
            setTimeout(() => {
                elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
            }, 100);
            
            state.messageCount = data.messages.length;
            if (elements.messageCountDisplay) {
                elements.messageCountDisplay.textContent = state.messageCount;
            }
            
            // Update conversation title with first message
            updateConversationTitle(data.messages[0]?.content || 'New Chat');
        } else {
            // No history, show welcome message
            if (!welcomeMsg) {
                showWelcomeMessage();
            }
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        // Show welcome message on error
        if (elements.chatMessages.children.length === 0) {
            showWelcomeMessage();
        }
    }
}

// Show welcome message
function showWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'welcome-message';
    welcomeDiv.innerHTML = `
        <i class="fas fa-robot"></i>
        <h2>Welcome to REFLEX</h2>
        <p>I'm an AI agent that learns from feedback using reinforcement learning. Ask me anything!</p>
        <div class="feature-pills">
            <span class="pill"><i class="fas fa-search"></i> Web Search</span>
            <span class="pill"><i class="fas fa-database"></i> RAG</span>
            <span class="pill"><i class="fas fa-brain"></i> Self-Improvement</span>
            <span class="pill"><i class="fas fa-memory"></i> Memory</span>
        </div>
    `;
    elements.chatMessages.appendChild(welcomeDiv);
}

// Update conversation title in sidebar
function updateConversationTitle(firstMessage) {
    const item = document.querySelector(`[data-session-id="${state.sessionId}"]`);
    if (item) {
        const titleEl = item.querySelector('.conversation-title');
        if (titleEl) {
            // Use first 30 chars of first message as title
            const title = firstMessage.length > 30 
                ? firstMessage.substring(0, 30) + '...'
                : firstMessage;
            titleEl.textContent = escapeHtml(title);
        }
    }
}

// Update conversation in sidebar (message count, last activity)
function updateConversationInSidebar() {
    const item = document.querySelector(`[data-session-id="${state.sessionId}"]`);
    if (item) {
        const metaEl = item.querySelector('.conversation-meta');
        if (metaEl) {
            metaEl.innerHTML = `
                <span>${state.messageCount} messages</span>
                <span>•</span>
                <span>Just now</span>
            `;
        }
    }
}

// Send message to agent with streaming
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message) return;
    
    // Disable input
    elements.sendBtn.disabled = true;
    elements.messageInput.disabled = true;
    
    // Add user message
    addMessage('user', message);
    
    // Clear input
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    
    // Show loading with activity updates
    const loadingId = addLoadingMessage('Analyzing your question...');
    
    // Create agent message container for streaming
    let agentMessageId = null;
    let accumulatedContent = '';
    let toolsUsed = [];
    let relevantSkills = [];
    
    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: state.sessionId,
                user_id: state.userId
            })
        });
        
        if (!response.ok) throw new Error('Failed to get response');
        
        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'status') {
                            // Update loading status
                            updateLoadingStatus(loadingId, data.message);
                        } else if (data.type === 'content') {
                            // Remove loading message on first content
                            if (!agentMessageId) {
                                removeLoadingMessage(loadingId);
                                agentMessageId = addStreamingMessage('agent', '');
                            }
                            
                            // Accumulate content
                            accumulatedContent += data.content;
                            
                            // Update streaming message
                            updateStreamingMessage(agentMessageId, accumulatedContent);
                        } else if (data.type === 'done') {
                            // Finalize message
                            toolsUsed = data.tools_used || [];
                            relevantSkills = data.relevant_skills || [];
                            const sources = data.sources || [];
                            
                            // Use accumulated content from done event if available, otherwise use our accumulated
                            const finalContent = data.accumulated || accumulatedContent;
                            
                            // Replace streaming message with final message
                            if (agentMessageId) {
                                replaceStreamingMessage(agentMessageId, finalContent, {
                                    tools: toolsUsed,
                                    skills: relevantSkills,
                                    sources: sources
                                });
                            } else {
                                // Fallback: create message if streaming didn't work
                                removeLoadingMessage(loadingId);
                                addMessage('agent', finalContent, {
                                    tools: toolsUsed,
                                    skills: relevantSkills,
                                    sources: sources
                                });
                            }
                            
                            // Update trajectory
                            state.lastTrajectory = {
                                message: accumulatedContent,
                                tools_used: toolsUsed,
                                relevant_skills: relevantSkills,
                                sources: sources
                            };
                            updateTrajectoryInfo(state.lastTrajectory);
                            
                            // Update message count
                            state.messageCount++;
                            elements.messageCountDisplay.textContent = state.messageCount;
                            
                            // Update conversation title if this is the first message
                            if (state.messageCount === 1) {
                                updateConversationTitle(message);
                            }
                            
                            // Update conversation in sidebar
                            updateConversationInSidebar();
                            
                            // Update stats
                            updateStats();
                        } else if (data.type === 'error') {
                            // Handle error
                            removeLoadingMessage(loadingId);
                            if (agentMessageId) {
                                replaceStreamingMessage(agentMessageId, `Error: ${data.error}`, { error: true });
                            } else {
                                addMessage('agent', `Error: ${data.error}`, { error: true });
                            }
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse SSE data:', parseError, line);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingId);
        if (agentMessageId) {
            replaceStreamingMessage(agentMessageId, 'Sorry, I encountered an error. Please try again.', { error: true });
        } else {
            addMessage('agent', 'Sorry, I encountered an error. Please try again.', { error: true });
        }
    } finally {
        elements.sendBtn.disabled = false;
        elements.messageInput.disabled = false;
        elements.messageInput.focus();
    }
}

// Add message to chat
function addMessage(type, content, meta = {}, shouldScroll = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    // Add fade-in animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(10px)';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatMessage(content);
    
    contentDiv.appendChild(textDiv);
    
    if (meta.tools || meta.skills || meta.sources) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        if (meta.tools && meta.tools.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-tools"></i> ${meta.tools.join(', ')}</span>`;
        }
        
        if (meta.skills && meta.skills.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-lightbulb"></i> ${meta.skills.join(', ')}</span>`;
        }
        
        if (meta.sources && meta.sources.length > 0) {
            const sourcesHtml = meta.sources.map(source => {
                const icon = source.type === 'rag' ? 'fa-database' : 'fa-globe';
                const identifier = source.identifier || (source.type === 'rag' ? 'RAG' : 'Web');
                const displayTitle = source.title || source.url || 'Unknown source';
                return `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" class="source-link" title="${escapeHtml(displayTitle)}">
                    <span class="source-identifier">[${identifier}]</span>
                    <i class="fas ${icon}"></i> ${escapeHtml(displayTitle)}
                </a>`;
            }).join('');
            metaDiv.innerHTML += `<div class="message-sources"><i class="fas fa-link"></i> Sources: ${sourcesHtml}</div>`;
        }
        
        contentDiv.appendChild(metaDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    
    // Animate in
    requestAnimationFrame(() => {
        messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    });
    
    if (shouldScroll) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

// Format message with proper markdown rendering
function formatMessage(content) {
    if (!content) return '';
    
    // Ensure content is a string
    const text = String(content);
    
    if (typeof marked !== 'undefined') {
        try {
            // Use marked.js for proper markdown rendering
            // Configure marked to be safe and handle breaks
            const markedOptions = {
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            };
            
            // Parse markdown
            const html = marked.parse(text, markedOptions);
            return html;
        } catch (error) {
            console.warn('Markdown parsing error:', error);
            // Fallback to basic formatting if marked fails
            return escapeHtml(text)
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }
    } else {
        // Fallback to basic formatting
        return escapeHtml(text)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
}

// Add loading message with activity status
function addLoadingMessage(statusText = 'Thinking...') {
    const id = `loading_${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent loading';
    messageDiv.id = id;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="agent-activity">
            <div class="activity-status">${statusText}</div>
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return id;
}

// Update loading message status
function updateLoadingStatus(id, statusText) {
    const loadingMsg = document.getElementById(id);
    if (loadingMsg) {
        const statusEl = loadingMsg.querySelector('.activity-status');
        if (statusEl) {
            statusEl.textContent = statusText;
            // Add animation
            statusEl.classList.add('pulse');
            setTimeout(() => statusEl.classList.remove('pulse'), 500);
        }
    }
}

// Remove loading message
function removeLoadingMessage(id) {
    const loadingMsg = document.getElementById(id);
    if (loadingMsg) loadingMsg.remove();
}

// Add streaming message (for real-time updates)
function addStreamingMessage(type, initialContent) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type} streaming`;
    const id = `streaming_${Date.now()}`;
    messageDiv.id = id;
    
    // Add fade-in animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(10px)';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text streaming-text';
    // Don't use formatMessage here - use plain text for streaming
    textDiv.textContent = initialContent || '';
    
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    
    // Animate in
    requestAnimationFrame(() => {
        messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    });
    
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return id;
}

// Update streaming message content
function updateStreamingMessage(id, content) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        const textDiv = messageDiv.querySelector('.streaming-text');
        if (textDiv) {
            // During streaming, render as plain text with basic formatting
            // This prevents markdown rendering issues with incomplete content
            // Full markdown will be rendered when streaming completes
            const escaped = escapeHtml(content);
            const basicFormatted = escaped
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
            textDiv.innerHTML = basicFormatted;
            
            // Auto-scroll to bottom
            elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        }
    }
}

// Replace streaming message with final message (adds metadata)
function replaceStreamingMessage(id, content, meta = {}) {
    const messageDiv = document.getElementById(id);
    if (!messageDiv) return;
    
    // Remove streaming class
    messageDiv.classList.remove('streaming');
    
    // Update content
    const textDiv = messageDiv.querySelector('.streaming-text');
    if (textDiv) {
        textDiv.classList.remove('streaming-text');
        textDiv.innerHTML = formatMessage(content);
    }
    
    // Add metadata if provided
    const contentDiv = messageDiv.querySelector('.message-content');
    if (contentDiv && (meta.tools || meta.skills || meta.sources)) {
        // Remove existing meta if any
        const existingMeta = contentDiv.querySelector('.message-meta');
        if (existingMeta) existingMeta.remove();
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        if (meta.tools && meta.tools.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-tools"></i> ${meta.tools.join(', ')}</span>`;
        }
        
        if (meta.skills && meta.skills.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-lightbulb"></i> ${meta.skills.join(', ')}</span>`;
        }
        
        if (meta.sources && meta.sources.length > 0) {
            const sourcesHtml = meta.sources.map(source => {
                const icon = source.type === 'rag' ? 'fa-database' : 'fa-globe';
                const identifier = source.identifier || (source.type === 'rag' ? 'RAG' : 'Web');
                const displayTitle = source.title || source.url || 'Unknown source';
                return `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" class="source-link" title="${escapeHtml(displayTitle)}">
                    <span class="source-identifier">[${identifier}]</span>
                    <i class="fas ${icon}"></i> ${escapeHtml(displayTitle)}
                </a>`;
            }).join('');
            metaDiv.innerHTML += `<div class="message-sources"><i class="fas fa-link"></i> Sources: ${sourcesHtml}</div>`;
        }
        
        contentDiv.appendChild(metaDiv);
    }
    
    // Scroll to bottom
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Update stats
async function updateStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        
        const stats = await response.json();
        
        elements.totalTasks.textContent = stats.total_tasks;
        elements.skillCount.textContent = stats.skill_count;
        elements.avgReward.textContent = stats.average_reward.toFixed(2);
        
        const successRate = stats.total_tasks > 0 
            ? ((stats.successful_tasks / stats.total_tasks) * 100).toFixed(1)
            : 0;
        elements.successRate.textContent = `${successRate}%`;
        
        // Update top skills
        if (stats.top_skills && stats.top_skills.length > 0) {
            elements.topSkillsList.innerHTML = stats.top_skills.map(skill => `
                <div class="skill-item">
                    <div class="skill-name">${skill.name}</div>
                    <div class="skill-meta">
                        <span>${(skill.success_rate * 100).toFixed(0)}% success</span>
                        <span>${skill.usage} uses</span>
                    </div>
                </div>
            `).join('');
        } else {
            elements.topSkillsList.innerHTML = '<p class="empty-state">No skills learned yet</p>';
        }
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Update trajectory info
function updateTrajectoryInfo(data) {
    // Tools used
    if (data.tools_used && data.tools_used.length > 0) {
        elements.toolsUsed.innerHTML = data.tools_used.map(tool => `
            <div class="info-item">${tool}</div>
        `).join('');
    } else {
        elements.toolsUsed.innerHTML = '<p class="empty-state">No tools used</p>';
    }
    
    // Skills applied
    if (data.relevant_skills && data.relevant_skills.length > 0) {
        elements.skillsApplied.innerHTML = data.relevant_skills.map(skill => `
            <div class="info-item">${skill}</div>
        `).join('');
    } else {
        elements.skillsApplied.innerHTML = '<p class="empty-state">No skills applied</p>';
    }
}

// Submit feedback
async function submitFeedback() {
    if (!state.lastTrajectory) {
        showFeedbackStatus('Please send a message first', 'error');
        return;
    }
    
    const feedbackData = {
        session_id: state.sessionId,
        task_success: parseFloat(elements.taskSuccess.value),
        quality_score: parseFloat(elements.quality.value),
        efficiency_score: parseFloat(elements.efficiency.value),
        user_feedback: parseFloat(elements.userFeedback.value)
    };
    
    // Add skill if checkbox is checked
    if (elements.createSkill.checked) {
        const skillName = document.getElementById('skillName').value.trim();
        const skillDesc = document.getElementById('skillDescription').value.trim();
        const skillContext = document.getElementById('skillContext').value.trim();
        
        if (!skillName || !skillDesc || !skillContext) {
            showFeedbackStatus('Please fill in all skill fields', 'error');
            return;
        }
        
        feedbackData.learned_skill = {
            name: skillName,
            description: skillDesc,
            context: skillContext,
            success_rate: feedbackData.task_success
        };
    }
    
    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(feedbackData)
        });
        
        if (!response.ok) throw new Error('Failed to submit feedback');
        
        const result = await response.json();
        
        showFeedbackStatus(
            `Feedback submitted! Reward: ${result.total_reward.toFixed(2)}`,
            'success'
        );
        
        // Reset form
        elements.taskSuccess.value = 0.5;
        elements.quality.value = 0.5;
        elements.efficiency.value = 0.5;
        elements.userFeedback.value = 0;
        elements.createSkill.checked = false;
        elements.skillForm.style.display = 'none';
        
        // Update values
        elements.taskSuccessValue.textContent = '0.5';
        elements.qualityValue.textContent = '0.5';
        elements.efficiencyValue.textContent = '0.5';
        elements.userFeedbackValue.textContent = '0.0';
        
        // Update stats
        updateStats();
        
    } catch (error) {
        console.error('Error submitting feedback:', error);
        showFeedbackStatus('Failed to submit feedback', 'error');
    }
}

// Show feedback status
function showFeedbackStatus(message, type) {
    elements.feedbackStatus.textContent = message;
    elements.feedbackStatus.className = `status-message ${type}`;
    
    setTimeout(() => {
        elements.feedbackStatus.className = 'status-message';
    }, 3000);
}

// Trigger training
async function triggerTraining() {
    elements.trainBtn.disabled = true;
    elements.trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch_size: 32 })
        });
        
        if (!response.ok) throw new Error('Training failed');
        
        const result = await response.json();
        
        // Update stats
        updateStats();
        
        // Show success
        alert('Training iteration completed successfully!');
        
    } catch (error) {
        console.error('Training error:', error);
        alert('Training failed: ' + error.message);
    } finally {
        elements.trainBtn.disabled = false;
        elements.trainBtn.innerHTML = '<i class="fas fa-graduation-cap"></i> Trigger Training';
    }
}

// Open skills modal
async function openSkillsModal() {
    elements.skillsModal.classList.add('active');
    elements.allSkillsList.innerHTML = '<p class="loading">Loading skills...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/skills?limit=100`);
        if (!response.ok) throw new Error('Failed to fetch skills');
        
        const data = await response.json();
        
        if (data.skills.length === 0) {
            elements.allSkillsList.innerHTML = '<p class="empty-state">No skills learned yet</p>';
            return;
        }
        
        renderSkills(data.skills);
        
    } catch (error) {
        console.error('Error loading skills:', error);
        elements.allSkillsList.innerHTML = '<p class="empty-state">Error loading skills</p>';
    }
}

// Render skills in modal
function renderSkills(skills) {
    elements.allSkillsList.innerHTML = skills.map(skill => `
        <div class="skill-card">
            <div class="skill-card-header">
                <div class="skill-card-title">${skill.name}</div>
                <div class="skill-badge">${(skill.success_rate * 100).toFixed(0)}%</div>
            </div>
            <div class="skill-card-description">${skill.description}</div>
            <div class="skill-card-stats">
                <div class="skill-stat">
                    <i class="fas fa-chart-line"></i>
                    <span><strong>${skill.usage_count}</strong> uses</span>
                </div>
                <div class="skill-stat">
                    <i class="fas fa-star"></i>
                    <span><strong>${skill.average_reward.toFixed(2)}</strong> avg reward</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Filter skills
function filterSkills(query) {
    const cards = elements.allSkillsList.querySelectorAll('.skill-card');
    const lowerQuery = query.toLowerCase();
    
    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(lowerQuery) ? 'block' : 'none';
    });
}

// Close skills modal
function closeSkillsModal() {
    elements.skillsModal.classList.remove('active');
    elements.skillSearch.value = '';
}

// Switch tabs
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === `${tabName}Tab`);
    });
}

// Toggle theme
function toggleTheme() {
    state.currentTheme = state.currentTheme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', state.currentTheme);
    localStorage.setItem('theme', state.currentTheme);
    updateThemeIcon();
}

// Update theme icon
function updateThemeIcon() {
    const icon = elements.themeToggle.querySelector('i');
    icon.className = state.currentTheme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Knowledge Base Management
async function loadKnowledgeBase() {
    try {
        const response = await fetch(`${API_BASE}/knowledge`);
        if (!response.ok) {
            if (elements.knowledgeStatus) {
                updateKnowledgeStatus(false, 'Knowledge base not available');
            }
            if (elements.knowledgeSourcesList) {
                elements.knowledgeSourcesList.innerHTML = '<p class="empty-state">Knowledge base not configured</p>';
            }
            return;
        }
        
        const data = await response.json();
        if (elements.knowledgeStatus) {
            updateKnowledgeStatus(data.enabled, data.enabled ? 'Active' : 'Disabled');
        }
        if (elements.knowledgeSourcesList) {
            renderKnowledgeSources(data.urls || []);
        }
    } catch (error) {
        console.error('Error loading knowledge base:', error);
        if (elements.knowledgeStatus) {
            updateKnowledgeStatus(false, 'Error loading knowledge base');
        }
        if (elements.knowledgeSourcesList) {
            elements.knowledgeSourcesList.innerHTML = '<p class="empty-state">Error loading knowledge base</p>';
        }
    }
}

function updateKnowledgeStatus(enabled, message) {
    if (!elements.knowledgeStatus) return;
    
    const indicator = elements.knowledgeStatus.querySelector('.status-indicator');
    if (indicator) {
        indicator.innerHTML = `
            <i class="fas fa-circle ${enabled ? 'active' : ''}"></i>
            <span>${message}</span>
        `;
    }
}

function renderKnowledgeSources(urls) {
    if (!elements.knowledgeSourcesList) return;
    
    if (urls.length === 0) {
        elements.knowledgeSourcesList.innerHTML = '<p class="empty-state">No knowledge sources added yet</p>';
        return;
    }
    
    elements.knowledgeSourcesList.innerHTML = urls.map((url) => `
        <div class="source-item">
            <div class="source-content">
                <i class="fas fa-link"></i>
                <span class="source-url">${escapeHtml(url)}</span>
            </div>
            <button class="source-action-btn" onclick="removeKnowledgeSource('${escapeHtml(url)}')" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

async function addKnowledgeSource() {
    const url = elements.knowledgeUrl?.value.trim();
    if (!url) {
        showNotification('Please enter a valid URL', 'error');
        return;
    }
    
    try {
        elements.addKnowledgeBtn.disabled = true;
        elements.addKnowledgeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
        
        const response = await fetch(`${API_BASE}/knowledge/urls`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            let errorMessage = 'Failed to add URL';
            try {
                const error = await response.json();
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        elements.knowledgeUrl.value = '';
        await loadKnowledgeBase();
        showNotification(result.message || 'Knowledge source added successfully!', 'success');
        
    } catch (error) {
        console.error('Error adding knowledge source:', error);
        showNotification(error.message || 'Failed to add knowledge source', 'error');
    } finally {
        elements.addKnowledgeBtn.disabled = false;
        elements.addKnowledgeBtn.innerHTML = '<i class="fas fa-plus"></i> Add URL';
    }
}

async function removeKnowledgeSource(url) {
    if (!confirm(`Are you sure you want to remove this knowledge source?\n${url}`)) return;
    
    try {
        const response = await fetch(`${API_BASE}/knowledge/urls`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            let errorMessage = 'Failed to remove URL';
            try {
                const error = await response.json();
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        await loadKnowledgeBase();
        showNotification(result.message || 'Knowledge source removed', 'success');
        
    } catch (error) {
        console.error('Error removing knowledge source:', error);
        showNotification(error.message || 'Failed to remove knowledge source', 'error');
    }
}

async function reloadKnowledgeBase() {
    if (!confirm('This will reload all knowledge sources. This may take a while. Continue?')) return;
    
    try {
        elements.reloadKnowledgeBtn.disabled = true;
        elements.reloadKnowledgeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Reloading...';
        
        const response = await fetch(`${API_BASE}/knowledge/reload`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            // Try to get error message from response
            let errorMessage = 'Failed to reload knowledge base';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                // If response is not JSON, use status text
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        showNotification(result.message || 'Knowledge base reloaded successfully!', 'success');
        await loadKnowledgeBase();
        
    } catch (error) {
        console.error('Error reloading knowledge base:', error);
        showNotification(error.message || 'Failed to reload knowledge base', 'error');
    } finally {
        elements.reloadKnowledgeBtn.disabled = false;
        elements.reloadKnowledgeBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Reload Knowledge Base';
    }
}

async function clearKnowledgeBase() {
    if (!confirm('Are you sure you want to clear all knowledge sources? This cannot be undone.')) return;
    
    try {
        elements.clearKnowledgeBtn.disabled = true;
        elements.clearKnowledgeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
        
        const response = await fetch(`${API_BASE}/knowledge/clear`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            let errorMessage = 'Failed to clear knowledge base';
            try {
                const error = await response.json();
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        showNotification(result.message || 'Knowledge base cleared', 'success');
        await loadKnowledgeBase();
        
    } catch (error) {
        console.error('Error clearing knowledge base:', error);
        showNotification(error.message || 'Failed to clear knowledge base', 'error');
    } finally {
        elements.clearKnowledgeBtn.disabled = false;
        elements.clearKnowledgeBtn.innerHTML = '<i class="fas fa-trash"></i> Clear All';
    }
}

// Make removeKnowledgeSource available globally
window.removeKnowledgeSource = removeKnowledgeSource;

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

