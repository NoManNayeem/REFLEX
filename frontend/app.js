// REFLEX - Research Engine with Feedback-Driven Learning - Frontend Application

// API_BASE - Use localhost for browser access (works in both Docker and local dev)
// In Docker, the browser accesses localhost:8000 which is mapped to the backend container
const API_BASE = window.API_BASE_URL || 'http://localhost:8000/api';
console.log('API Base URL:', API_BASE);

// State management
const state = {
    sessionId: `session_${Date.now()}`,
    userId: 'demo_user',
    messageCount: 0,
    lastTrajectory: null,
    currentTheme: localStorage.getItem('theme') || 'light'
};

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
    skillSearch: document.getElementById('skillSearch')
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
    
    console.log('✅ Application initialized');
}

// Send message to agent
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
    
    // Show loading
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: state.sessionId,
                user_id: state.userId
            })
        });
        
        if (!response.ok) throw new Error('Failed to get response');
        
        const data = await response.json();
        
        // Remove loading
        removeLoadingMessage(loadingId);
        
        // Add agent message
        addMessage('agent', data.message, {
            tools: data.tools_used,
            skills: data.relevant_skills
        });
        
        // Update trajectory
        state.lastTrajectory = data;
        updateTrajectoryInfo(data);
        
        // Update message count
        state.messageCount++;
        elements.messageCountDisplay.textContent = state.messageCount;
        
        // Update stats
        updateStats();
        
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingId);
        addMessage('agent', 'Sorry, I encountered an error. Please try again.', { error: true });
    } finally {
        elements.sendBtn.disabled = false;
        elements.messageInput.disabled = false;
        elements.messageInput.focus();
    }
}

// Add message to chat
function addMessage(type, content, meta = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatMessage(content);
    
    contentDiv.appendChild(textDiv);
    
    if (meta.tools || meta.skills) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        if (meta.tools && meta.tools.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-tools"></i> ${meta.tools.join(', ')}</span>`;
        }
        
        if (meta.skills && meta.skills.length > 0) {
            metaDiv.innerHTML += `<span><i class="fas fa-lightbulb"></i> ${meta.skills.join(', ')}</span>`;
        }
        
        contentDiv.appendChild(metaDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// Format message (basic markdown support)
function formatMessage(content) {
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

// Add loading message
function addLoadingMessage() {
    const id = `loading_${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.id = id;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return id;
}

// Remove loading message
function removeLoadingMessage(id) {
    const loadingMsg = document.getElementById(id);
    if (loadingMsg) loadingMsg.remove();
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

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

