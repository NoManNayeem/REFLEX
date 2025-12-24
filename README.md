# 🤖 REFLEX - Research Engine with Feedback-Driven Learning

A functional proof-of-concept demonstrating a self-improving AI agent using **Reinforcement Learning**, **RAG (Retrieval Augmented Generation)**, and **Agno v2** framework with a modern web interface.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Agno](https://img.shields.io/badge/agno-v2.3.12-purple)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### 🧠 Agent Capabilities
- **Self-Improvement with RL**: Learns from feedback using GRPO-style reinforcement learning
- **Skill Library**: Accumulates and reuses learned skills across tasks
- **RAG with Vector DB**: LanceDB-powered knowledge retrieval
- **Web Search**: DuckDuckGo integration for current information
- **Persistent Memory**: SQLite storage for conversation history
- **Session Management**: Multi-user, multi-session support

### 🎨 Modern UI/UX
- **Real-time Chat**: Streaming responses with live updates
- **Training Dashboard**: Visual metrics and performance tracking
- **Skill Visualization**: Interactive skill library browser
- **Feedback System**: Intuitive reward mechanism
- **Dark/Light Mode**: Theme toggle with localStorage persistence
- **Responsive Design**: Works on desktop and mobile

### 🔧 Technical Stack
- **Backend**: FastAPI + Python 3.10+
- **Agent Framework**: Agno v2 (Dec 2025)
- **LLM**: Claude Sonnet 4
- **Vector DB**: LanceDB with hybrid search
- **Storage**: SQLite for sessions and memory
- **Frontend**: Pure HTML/CSS/JavaScript (no framework overhead)

## 🚀 Quick Start

### Prerequisites

```bash
# Required
- Python 3.10 or higher
- pip (Python package manager)
- API Keys:
  - Anthropic API key (for Claude)
  - OpenAI API key (for embeddings)

# Optional but recommended
- Git
- Virtual environment tool
```

### Installation

#### 1. Clone or Download

```bash
# Option A: Clone repository
git clone <your-repo-url>
cd REFLEX

# Option B: Download and extract ZIP
# Then navigate to the directory
```

#### 2. Project Structure

The project structure is already set up:

```
REFLEX/
├── backend/
│   ├── main.py
│   ├── agent_core.py
│   ├── models.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/
│   ├── skills/
│   └── db/
├── run.sh
└── README.md
```

#### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**backend/requirements.txt**:
```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
agno==2.3.12
anthropic==0.39.0
openai==1.54.0
duckduckgo-search==6.3.5
lancedb==0.16.0
numpy==1.26.4
python-dotenv==1.0.1
pydantic==2.10.2
websockets==14.1
```

#### 4. Configure Environment

Create `backend/.env`:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Get API Keys:**
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys

#### 5. Run the Application

**Option A: Using run script (macOS/Linux)**

```bash
chmod +x run.sh
./run.sh
```

**Option B: Manual start**

Terminal 1 (Backend):
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (Frontend):
```bash
cd frontend
python3 -m http.server 3000
# Or use any other static file server
```

#### 6. Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8000/api/health

## 📖 Usage Guide

### Basic Workflow

1. **Ask Questions**
   - Type your research question in the chat input
   - Examples:
     - "What is GRPO in reinforcement learning?"
     - "Explain self-improving AI agents"
     - "Compare GSPO and PPO algorithms"

2. **Receive Responses**
   - Agent searches web and knowledge base
   - Applies relevant learned skills
   - Streams response in real-time
   - Shows tools and skills used

3. **Provide Feedback**
   - Use sliders to rate response quality
   - Optionally create a skill from the interaction
   - Submit feedback to improve the agent

4. **Monitor Progress**
   - View live training metrics in sidebar
   - Check top performing skills
   - Track success rate and rewards

5. **Trigger Training**
   - Click "Trigger Training" to run RL update
   - Agent improves based on collected feedback
   - Skills are updated with new success rates

### Feedback System

**Sliders:**
- **Task Success** (0-1): How well did the agent complete the task?
- **Quality** (0-1): How good was the response quality?
- **Efficiency** (0-1): How efficient was the agent?
- **User Feedback** (-1 to 1): Your overall satisfaction

**Creating Skills:**
1. Check "Create Skill" checkbox
2. Fill in:
   - **Skill Name**: Short identifier (e.g., "grpo_research")
   - **Description**: What the skill does
   - **Context**: How to apply the skill
3. Submit feedback

**Skill Example:**
```
Name: reinforcement_learning_research
Description: Research RL algorithms and compare approaches
Context: Use web search for recent papers, compare algorithmic details
```

### API Usage

**Chat Request:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is GRPO?",
    "session_id": "session_1",
    "user_id": "user_123"
  }'
```

**Submit Feedback:**
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_1",
    "task_success": 0.9,
    "quality_score": 0.85,
    "efficiency_score": 0.8,
    "user_feedback": 1.0
  }'
```

**View Stats:**
```bash
curl http://localhost:8000/api/stats
```

**List Skills:**
```bash
curl http://localhost:8000/api/skills
```

## 🎯 Key Components

### Backend Architecture

```
┌─────────────────────────────────────┐
│        FastAPI Application          │
│  (main.py - API endpoints)          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   SelfImprovingResearchAgent        │
│  (agent_core.py - Core logic)       │
├─────────────────────────────────────┤
│  • Agno Agent (Claude Sonnet 4)     │
│  • SkillLibrary (persistence)       │
│  • TrajectoryBuffer (RL memory)     │
│  • RewardSignal (feedback)          │
└─────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│         Storage Layer               │
├─────────────────────────────────────┤
│  • SQLite (sessions, memory)        │
│  • LanceDB (vector embeddings)     │
│  • JSON (skill library)             │
└─────────────────────────────────────┘
```

### Frontend Architecture

```
┌─────────────────────────────────────┐
│          index.html                 │
│  (Structure and layout)             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│          styles.css                 │
│  (Modern styling + themes)          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│           app.js                    │
│  (Application logic)                │
├─────────────────────────────────────┤
│  • State management                 │
│  • API communication                │
│  • UI updates                       │
│  • Event handling                   │
└─────────────────────────────────────┘
```

## 🔬 How It Works

### 1. Self-Improvement Loop

```
User Query → Agent Response → Feedback → Skill Update → Improved Performance
     ↑                                                           │
     └───────────────────────────────────────────────────────────┘
```

### 2. RL Training Process

1. **Trajectory Collection**: Store (query, response, reward) tuples
2. **Advantage Computation**: Use GRPO-style group-relative advantages
3. **Skill Update**: Boost skills with positive advantages
4. **Agent Recreation**: Rebuild agent with updated skill context

### 3. RAG Pipeline

1. **Indexing**: Documents → Chunks → Embeddings → Vector DB
2. **Query**: User question → Embedding
3. **Retrieval**: Semantic search → Top-k relevant chunks
4. **Generation**: Context + Query → LLM → Response

### 4. Skill System

- **Extraction**: Successful interactions → Named skills
- **Storage**: Persistent JSON with metadata
- **Retrieval**: Keyword matching + success rate ranking
- **Application**: Add skill context to agent instructions
- **Evolution**: Update success rates based on outcomes

## 🎨 UI Features

### Chat Interface
- ✅ Real-time message streaming
- ✅ Markdown formatting
- ✅ Tool call visibility
- ✅ Skill usage display
- ✅ Loading indicators
- ✅ Auto-scroll

### Training Dashboard
- ✅ Total tasks counter
- ✅ Success rate percentage
- ✅ Average reward display
- ✅ Skill count tracker
- ✅ Top skills list
- ✅ Live updates

### Feedback Panel
- ✅ Multi-dimensional rating sliders
- ✅ Real-time value display
- ✅ Skill creation form
- ✅ Submit confirmation
- ✅ Status messages

### Skills Modal
- ✅ Searchable skill list
- ✅ Detailed skill cards
- ✅ Success rate badges
- ✅ Usage statistics
- ✅ Reward history

## 🐛 Troubleshooting

### Common Issues

**1. Agent not starting**
```bash
# Check API keys
cat backend/.env

# Verify installation
pip list | grep agno

# Check logs
cd backend
python -m main
```

**2. Knowledge base errors**
```
Solution: Comment out knowledge.load() after first run
Location: backend/agent_core.py line ~140
```

**3. Frontend not connecting**
```bash
# Verify backend is running
curl http://localhost:8000/api/health

# Check CORS settings
# Should see "allow_origins: ['*']" in backend/main.py

# Try different port
python3 -m http.server 8080
```

**4. Database locked errors**
```bash
# Clear databases
rm -rf data/db/*

# Restart application
./run.sh
```

**5. Module not found errors**
```bash
# Reinstall dependencies
cd backend
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Performance Issues

**Slow responses:**
- Reduce knowledge base size
- Use smaller embedding model
- Disable web search for testing
- Increase batch processing

**High memory usage:**
- Limit trajectory buffer size (line 51 in agent_core.py)
- Clear old sessions periodically
- Use PostgreSQL instead of SQLite for production

## 🚀 Production Deployment

### Backend

**1. Use Production Server**
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**2. Environment Variables**
```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DATABASE_URL=postgresql://...
```

**3. Database Migration**
Replace SQLite with PostgreSQL:
```python
from agno.db.postgres import PostgresDb

db = PostgresDb(
    table_name="agent_sessions",
    db_url=os.getenv("DATABASE_URL")
)
```

### Frontend

**1. Build for Production**
- Minify CSS/JS
- Enable compression
- Add caching headers
- Use CDN for static assets

**2. Deploy Options**
- Vercel (frontend)
- Railway (backend + DB)
- AWS (EC2 + RDS)
- Docker (containerized)

### Docker Deployment

**Dockerfile** (backend):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
  
  frontend:
    image: nginx:alpine
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
```

## 📊 Monitoring

### Logs

**Backend logs:**
```bash
tail -f backend/logs/app.log
```

**Access logs:**
```bash
tail -f backend/logs/access.log
```

### Metrics

**Key metrics to track:**
- Success rate trend
- Average reward over time
- Skill accumulation rate
- Response latency
- Error rate

**Prometheus integration:**
```python
from prometheus_client import Counter, Histogram

task_counter = Counter('agent_tasks_total', 'Total tasks')
reward_histogram = Histogram('agent_reward', 'Task rewards')
```

## 🧪 Testing

**Run tests:**
```bash
cd backend
pytest tests/
```

**Manual testing:**
```bash
# Health check
curl http://localhost:8000/api/health

# Chat test
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "session_id": "test"}'

# Stats test
curl http://localhost:8000/api/stats
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **Agno Team**: For the excellent agent framework
- **Anthropic**: For Claude API
- **DeepSeek/Qwen**: For RL research (GRPO/GSPO)
- **Community**: For feedback and contributions

## 📚 Resources

- [Agno Documentation](https://docs.agno.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Claude API Docs](https://docs.anthropic.com)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [SAGE Paper](https://arxiv.org/abs/2410.01952)

## 📧 Support

- Issues: [GitHub Issues](your-repo/issues)
- Discussions: [GitHub Discussions](your-repo/discussions)
- Email: support@example.com

---

**Built with ❤️ using Agno v2, FastAPI, and modern web technologies**

**Star ⭐ this repo if you find it useful!**

