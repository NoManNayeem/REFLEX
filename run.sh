#!/bin/bash

# REFLEX - Research Engine with Feedback-Driven Learning - Startup Script

echo "🚀 Starting REFLEX - Research Engine with Feedback-Driven Learning..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
    cd ..
else
    echo -e "${GREEN}Virtual environment found${NC}"
fi

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo -e "${YELLOW}Please create backend/.env with your API keys:${NC}"
    echo "ANTHROPIC_API_KEY=your_key_here"
    echo "OPENAI_API_KEY=your_key_here"
    exit 1
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/skills
mkdir -p data/db

# Start backend
echo -e "${GREEN}Starting FastAPI backend...${NC}"
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Backend is running at http://localhost:8000${NC}"
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Start frontend
echo -e "${GREEN}Starting frontend server...${NC}"
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

# Wait a moment
sleep 2

echo ""
echo -e "${GREEN}🎉 Application is ready!${NC}"
echo ""
echo "📍 Backend API:  http://localhost:8000"
echo "📍 Frontend UI:  http://localhost:3000"
echo "📍 API Docs:     http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ Servers stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Keep script running
wait

