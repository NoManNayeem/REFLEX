"""
FastAPI Backend for REFLEX - Research Engine with Feedback-Driven Learning
Handles API endpoints, streaming, and agent orchestration
"""

import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agent_core import SelfImprovingResearchAgent, RewardSignal, Skill
from models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SkillCreate,
    TrainingRequest,
    SessionInfo,
    AgentStats
)

# Load environment variables from root .env file
# Try root directory first, then fallback to backend/.env
import pathlib
root_env = pathlib.Path(__file__).parent.parent / ".env"
backend_env = pathlib.Path(__file__).parent / ".env"
if root_env.exists():
    load_dotenv(root_env)
    logger.info(f"Loaded .env from root directory: {root_env}")
elif backend_env.exists():
    load_dotenv(backend_env)
    logger.info(f"Loaded .env from backend directory: {backend_env}")
else:
    load_dotenv()  # Try default locations
    logger.warning("No .env file found, using system environment variables")

# Global agent instance
agent_instance: Optional[SelfImprovingResearchAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent_instance
    logger.info("🚀 Initializing REFLEX - Research Engine with Feedback-Driven Learning...")
    
    try:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        logger.debug(f"Anthropic API key present: {bool(anthropic_key)}")
        logger.debug(f"OpenAI API key present: {bool(openai_key)}")
        
        # Use absolute path for database
        import pathlib
        db_path = pathlib.Path(__file__).parent.parent / "data" / "db" / "agent.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        agent_instance = SelfImprovingResearchAgent(
            api_key=anthropic_key,
            openai_api_key=openai_key,
            db_path=str(db_path)
        )
        logger.info("✅ Agent initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("🛑 Shutting down agent...")


# Create FastAPI app
app = FastAPI(
    title="REFLEX API",
    description="Research Engine with Feedback-Driven Learning - API for interacting with an RL-based self-improving research agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_ready": agent_instance is not None
    }


@app.get("/api/agent/status")
async def agent_status():
    """Get agent status and configuration"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = agent_instance.get_stats()
    return {
        "status": "ready",
        "skills_count": stats['skill_count'],
        "trajectory_count": stats['trajectory_count'],
        "total_tasks": stats['training_stats']['total_tasks'],
        "success_rate": (
            stats['training_stats']['successful_tasks'] / 
            max(stats['training_stats']['total_tasks'], 1)
        )
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the agent and get a response
    """
    logger.info(f"Received chat request: session_id={request.session_id}, message_length={len(request.message)}")
    
    if not agent_instance:
        logger.error("Agent not initialized when chat request received")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.debug(f"Processing message: {request.message[:100]}...")
        result = agent_instance.run_task(
            query=request.message,
            session_id=request.session_id or "default",
            user_id=request.user_id or "default_user"
        )
        
        response_content = result['response'].content
        logger.info(f"Response generated: length={len(response_content)}, skills_used={len(result.get('relevant_skills', []))}")
        
        return ChatResponse(
            message=response_content,
            session_id=request.session_id or "default",
            trajectory_id=request.session_id or "default",
            tools_used=[
                tool.get("name", "unknown") 
                for tool in result['trajectory'].get('tools_used', [])
            ],
            relevant_skills=[
                skill.name for skill in result['relevant_skills']
            ],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/api/chat/stream")
async def chat_stream(
    message: str = Query(..., description="Message to send to agent"),
    session_id: str = Query("default", description="Session ID"),
    user_id: str = Query("default_user", description="User ID")
):
    """
    Stream agent response using Server-Sent Events (SSE)
    """
    logger.info(f"Received stream request: session_id={session_id}, message_length={len(message)}")
    
    if not agent_instance:
        logger.error("Agent not initialized when stream request received")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def event_generator():
        try:
            logger.debug(f"Starting stream for message: {message[:100]}...")
            # Run agent and get response
            result = agent_instance.run_task(
                query=message,
                session_id=session_id,
                user_id=user_id
            )
            
            response_content = result['response'].content
            
            # Stream response word by word for demo effect
            words = response_content.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "content": word + " ",
                    "done": i == len(words) - 1,
                    "tools_used": [
                        tool.get("name", "unknown") 
                        for tool in result['trajectory'].get('tools_used', [])
                    ] if i == len(words) - 1 else [],
                    "relevant_skills": [
                        skill.name for skill in result['relevant_skills']
                    ] if i == len(words) - 1 else []
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            # Send final message
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream generator: {str(e)}", exc_info=True)
            error_data = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a task to train the agent
    """
    logger.info(f"Received feedback: session_id={request.session_id}, task_success={request.task_success}, quality={request.quality_score}")
    
    if not agent_instance:
        logger.error("Agent not initialized when feedback received")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Create reward signal
        reward_signal = RewardSignal(
            task_success=request.task_success,
            quality_score=request.quality_score,
            efficiency_score=request.efficiency_score,
            user_feedback=request.user_feedback
        )
        
        # Create skill if provided
        learned_skill = None
        if request.learned_skill:
            learned_skill = Skill(
                name=request.learned_skill.name,
                description=request.learned_skill.description,
                context=request.learned_skill.context,
                success_rate=request.learned_skill.success_rate,
                usage_count=1,
                average_reward=reward_signal.compute_total_reward()
            )
        
        # Get trajectory (simplified - in real implementation, store trajectories)
        trajectory = {
            "query": "feedback_submission",
            "session_id": request.session_id,
            "relevant_skills": []
        }
        
        # Provide feedback to agent
        logger.debug(f"Processing feedback: reward={reward_signal.compute_total_reward()}, skill_added={learned_skill is not None}")
        agent_instance.provide_feedback(
            trajectory=trajectory,
            reward_signal=reward_signal,
            learned_skill=learned_skill
        )
        
        logger.info(f"Feedback processed successfully: total_reward={reward_signal.compute_total_reward()}")
        return {
            "status": "success",
            "message": "Feedback processed successfully",
            "total_reward": reward_signal.compute_total_reward(),
            "skill_added": learned_skill is not None
        }
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@app.get("/api/skills")
async def list_skills(
    limit: int = Query(50, description="Maximum number of skills to return"),
    search: Optional[str] = Query(None, description="Search query for skills")
):
    """
    List all learned skills
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        all_skills = list(agent_instance.skill_library.skills.values())
        
        # Filter by search if provided
        if search:
            search_lower = search.lower()
            all_skills = [
                s for s in all_skills 
                if search_lower in s.name.lower() or 
                   search_lower in s.description.lower()
            ]
        
        # Sort by success rate and usage
        all_skills.sort(
            key=lambda s: s.success_rate * (1 + s.usage_count),
            reverse=True
        )
        
        # Limit results
        skills = all_skills[:limit]
        
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "context": s.context,
                    "success_rate": s.success_rate,
                    "usage_count": s.usage_count,
                    "average_reward": s.average_reward
                }
                for s in skills
            ],
            "total_count": len(agent_instance.skill_library.skills),
            "filtered_count": len(all_skills)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skills error: {str(e)}")


@app.get("/api/skills/{skill_name}")
async def get_skill(skill_name: str):
    """
    Get details of a specific skill
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    skill = agent_instance.skill_library.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    
    return {
        "name": skill.name,
        "description": skill.description,
        "context": skill.context,
        "success_rate": skill.success_rate,
        "usage_count": skill.usage_count,
        "average_reward": skill.average_reward
    }


@app.post("/api/skills")
async def create_skill(skill: SkillCreate):
    """
    Manually add a skill to the library
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        new_skill = Skill(
            name=skill.name,
            description=skill.description,
            context=skill.context,
            success_rate=skill.success_rate,
            usage_count=0,
            average_reward=0.0
        )
        
        agent_instance.skill_library.add_skill(new_skill)
        
        return {
            "status": "success",
            "message": f"Skill '{skill.name}' added successfully",
            "skill": {
                "name": new_skill.name,
                "description": new_skill.description,
                "success_rate": new_skill.success_rate
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill creation error: {str(e)}")


@app.post("/api/train")
async def trigger_training(request: TrainingRequest):
    """
    Trigger a training iteration
    """
    logger.info(f"Training request received: batch_size={request.batch_size}")
    
    if not agent_instance:
        logger.error("Agent not initialized when training requested")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.debug("Starting training iteration...")
        agent_instance.train_iteration(batch_size=request.batch_size)
        
        stats = agent_instance.get_stats()
        logger.info(f"Training completed: total_tasks={stats['training_stats']['total_tasks']}, skill_count={stats['skill_count']}")
        
        return {
            "status": "success",
            "message": "Training iteration completed",
            "batch_size": request.batch_size,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/api/stats", response_model=AgentStats)
async def get_stats():
    """
    Get training statistics and agent metrics
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        stats = agent_instance.get_stats()
        
        return AgentStats(
            total_tasks=stats['training_stats']['total_tasks'],
            successful_tasks=stats['training_stats']['successful_tasks'],
            average_reward=stats['training_stats']['average_reward'],
            improvement_rate=stats['training_stats']['improvement_rate'],
            skill_count=stats['skill_count'],
            trajectory_count=stats['trajectory_count'],
            top_skills=stats['top_skills']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.get("/api/sessions")
async def list_sessions(limit: int = Query(20, description="Number of sessions")):
    """
    List recent sessions (simplified implementation)
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # This is simplified - in production, query database
    return {
        "sessions": [],
        "message": "Session listing not yet implemented in this POC"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

