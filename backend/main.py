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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import configuration and security
from config import settings
from security import SecurityValidator

# Configure logging with settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agent_core import SelfImprovingResearchAgent, RewardSignal, Skill
from db_helper import ConversationDB
from models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SkillCreate,
    TrainingRequest,
    SessionInfo,
    AgentStats,
    MessageUpdate
)

# Import knowledge base modules for status check
try:
    from agno.knowledge import Knowledge
except ImportError:
    Knowledge = None

try:
    from agno.vectordb.lancedb import LanceDb, SearchType
except ImportError:
    LanceDb = None
    SearchType = None

try:
    from agno.knowledge.embedder.openai import OpenAIEmbedder
except ImportError:
    OpenAIEmbedder = None

try:
    from agno.knowledge.reader.website_reader import WebsiteReader
except ImportError:
    WebsiteReader = None

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
    # Only warn if env vars are not already set (e.g., from Docker)
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No .env file found and no API keys in environment variables")
    else:
        logger.info("Using environment variables (no .env file needed)")

# Global agent instance
agent_instance: Optional[SelfImprovingResearchAgent] = None
conversation_db: Optional[ConversationDB] = None


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
        
        # Initialize conversation database
        global conversation_db
        conversation_db = ConversationDB(str(db_path))
        
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
    lifespan=lifespan,
    debug=settings.debug
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware with security
logger.info(f"CORS origins: {settings.cors_origins_list}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["Content-Type", "Authorization"],
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
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Send a message to the agent and get a response
    """
    # Validate input
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(chat_request.message) > settings.max_message_length:
        raise HTTPException(
            status_code=400, 
            detail=f"Message exceeds maximum length of {settings.max_message_length} characters"
        )
    
    # Validate session and user IDs if provided
    if chat_request.session_id and not SecurityValidator.validate_session_id(chat_request.session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    
    if chat_request.user_id and not SecurityValidator.validate_user_id(chat_request.user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    logger.info(f"Received chat request: session_id={chat_request.session_id}, message_length={len(chat_request.message)}")
    
    if not agent_instance:
        logger.error("Agent not initialized when chat request received")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        session_id = chat_request.session_id or "default"
        user_id = chat_request.user_id or "default_user"
        
        # Store user message
        if conversation_db:
            conversation_db.save_message(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content=chat_request.message
            )
        
        logger.debug(f"Processing message: {chat_request.message[:100]}...")
        result = agent_instance.run_task(
            query=chat_request.message,
            session_id=session_id,
            user_id=user_id
        )
        
        response_content = result['response'].content
        tools_used = [
            tool.get("name", "unknown") 
            for tool in result['trajectory'].get('tools_used', [])
        ]
        relevant_skills = [
            skill.name for skill in result['relevant_skills']
        ]
        
        logger.info(f"Response generated: length={len(response_content)}, skills_used={len(relevant_skills)}")
        
        # Store agent response
        if conversation_db:
            conversation_db.save_message(
                session_id=session_id,
                user_id=user_id,
                role="agent",
                content=response_content,
                tools_used=tools_used,
                skills_applied=relevant_skills,
                trajectory_id=session_id
            )
        
        return ChatResponse(
            message=response_content,
            session_id=session_id,
            trajectory_id=session_id,
            tools_used=tools_used,
            relevant_skills=relevant_skills,
            sources=result.get('sources', []),
            critic_score=result.get('critic_score', 0.0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/api/chat/stream")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat_stream(request: Request, chat_request: ChatRequest):
    """
    Stream agent response using Server-Sent Events (SSE)
    Uses Agno's streaming capabilities for real-time response generation
    """
    # Validate input
    if not chat_request.message or not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(chat_request.message) > settings.max_message_length:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds maximum length of {settings.max_message_length} characters"
        )
    
    # Validate session and user IDs
    if chat_request.session_id and not SecurityValidator.validate_session_id(chat_request.session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    
    if chat_request.user_id and not SecurityValidator.validate_user_id(chat_request.user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    logger.info(f"Received stream request: session_id={chat_request.session_id}, message_length={len(chat_request.message)}")
    
    if not agent_instance:
        logger.error("Agent not initialized when stream request received")
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    session_id = chat_request.session_id or "default"
    user_id = chat_request.user_id or "default_user"
    
    async def event_generator():
        full_response_content = ""
        tools_used = []
        relevant_skills = []
        run_response = None
        
        try:
            # Store user message
            if conversation_db:
                conversation_db.save_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=chat_request.message
                )
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': 'Analyzing your question...'})}\n\n"
            await asyncio.sleep(0.05)
            
            yield f"data: {json.dumps({'type': 'status', 'status': 'searching', 'message': 'Searching for information...'})}\n\n"
            await asyncio.sleep(0.05)
            
            logger.debug(f"Starting real stream for message: {chat_request.message[:100]}...")
            
            # Use the agent's streaming method
            async for chunk in agent_instance.run_task_stream(
                query=chat_request.message,
                session_id=session_id,
                user_id=user_id
            ):
                if chunk.get('type') == 'status':
                    # Stream status update (tool calls, searching, etc.)
                    status_data = {
                        'type': 'status',
                        'status': chunk.get('status', 'thinking'),
                        'message': chunk.get('message', 'Processing...')
                    }
                    yield f"data: {json.dumps(status_data)}\n\n"
                    
                elif chunk.get('type') == 'content':
                    # Stream content chunk
                    full_response_content += chunk.get('content', '')
                    chunk_data = {
                        'type': 'content',
                        'content': chunk.get('content', ''),
                        'done': False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                elif chunk.get('type') == 'done':
                    # Final chunk with metadata
                    full_response_content = chunk.get('accumulated', full_response_content)
                    run_response = chunk.get('full_response')
                    tools_used = chunk.get('tools_used', [])
                    relevant_skills = chunk.get('relevant_skills', [])
                    sources = chunk.get('sources', [])
                    critic_score = chunk.get('critic_score', 0.0)
                    
                    # Send final content update
                    done_data = {
                        'type': 'done',
                        'content': '',
                        'done': True,
                        'tools_used': tools_used,
                        'relevant_skills': relevant_skills,
                        'sources': sources,
                        'critic_score': critic_score
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
                    
                elif chunk.get('type') == 'error':
                    # Error occurred
                    error_msg = chunk.get('error', 'Unknown error')
                    error_data = {
                        'type': 'error',
                        'error': error_msg,
                        'done': True
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
            
            # Store agent response after streaming completes
            if conversation_db and full_response_content:
                conversation_db.save_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="agent",
                    content=full_response_content,
                    tools_used=tools_used,
                    skills_applied=relevant_skills,
                    trajectory_id=session_id
                )
                logger.info(f"Stored agent response: length={len(full_response_content)}")
            
        except Exception as e:
            logger.error(f"Error in stream generator: {str(e)}", exc_info=True)
            error_data = {
                "type": "error",
                "error": str(e),
                "done": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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
        # Create reward signal with adaptive weights if available
        reward_signal = RewardSignal(
            task_success=request.task_success,
            quality_score=request.quality_score,
            efficiency_score=request.efficiency_score,
            user_feedback=request.user_feedback,
            adaptive_weights=getattr(agent_instance, 'adaptive_reward_weights', None)
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


@app.get("/api/chat/history")
async def get_chat_history(
    session_id: str = Query(..., description="Session ID"),
    limit: Optional[int] = Query(None, description="Maximum number of messages")
):
    """
    Get conversation history for a session
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    try:
        messages = conversation_db.get_conversation_history(session_id, limit)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")


@app.get("/api/sessions")
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(20, description="Number of sessions")
):
    """
    List recent sessions
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    try:
        sessions = conversation_db.get_all_sessions(user_id)
        return {
            "sessions": sessions[:limit],
            "total": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Sessions error: {str(e)}")


@app.get("/api/chat/messages/{message_id}")
async def get_message(message_id: int):
    """
    Get a specific message by ID
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    try:
        message = conversation_db.get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        return message
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving message: {e}")
        raise HTTPException(status_code=500, detail=f"Message error: {str(e)}")


@app.put("/api/chat/messages/{message_id}")
async def update_message(message_id: int, update: MessageUpdate):
    """
    Update a message
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    try:
        success = conversation_db.update_message(
            message_id=message_id,
            content=update.content,
            tools_used=update.tools_used,
            skills_applied=update.skills_applied
        )
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        
        # Return updated message
        updated_message = conversation_db.get_message(message_id)
        return {
            "status": "success",
            "message": "Message updated successfully",
            "data": updated_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating message: {e}")
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")


@app.delete("/api/chat/messages/{message_id}")
async def delete_message(message_id: int):
    """
    Delete a specific message by ID
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    try:
        success = conversation_db.delete_message(message_id)
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return {
            "status": "success",
            "message": f"Message {message_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting message: {e}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete all messages for a session
    """
    if not conversation_db:
        raise HTTPException(status_code=503, detail="Conversation database not initialized")
    
    # Validate session ID
    if not SecurityValidator.validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    
    try:
        success = conversation_db.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or already empty")
        
        return {
            "status": "success",
            "message": f"Session {session_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


# Knowledge Base Management Endpoints
@app.get("/api/knowledge")
async def get_knowledge_base():
    """
    Get knowledge base status and URLs
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Check if knowledge base is enabled (has OpenAI key and modules)
        has_openai_key = bool(agent_instance.openai_api_key)
        has_modules = bool(Knowledge and LanceDb and OpenAIEmbedder and WebsiteReader)
        enabled = agent_instance.knowledge is not None and has_openai_key and has_modules
        urls = agent_instance.knowledge_urls if agent_instance.knowledge_urls else []
        
        return {
            "enabled": enabled,
            "urls": urls,
            "count": len(urls),
            "has_openai_key": has_openai_key,
            "has_modules": has_modules
        }
    except Exception as e:
        logger.error(f"Error getting knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge base error: {str(e)}")


@app.post("/api/knowledge/urls")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def add_knowledge_url(request: Request, url_request: Dict[str, str]):
    """
    Add a URL to the knowledge base
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    url = url_request.get('url')
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # Validate URL
    if not SecurityValidator.validate_url(url):
        raise HTTPException(status_code=400, detail="Invalid or unsafe URL")
    
    try:
        success = agent_instance.add_knowledge_url(url)
        if success:
            return {
                "status": "success",
                "message": f"URL added successfully: {url}",
                "urls": agent_instance.knowledge_urls
            }
        else:
            raise HTTPException(status_code=400, detail="URL already exists or failed to add")
    except Exception as e:
        logger.error(f"Error adding knowledge URL: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding URL: {str(e)}")


@app.delete("/api/knowledge/urls")
async def remove_knowledge_url(request: Dict[str, str]):
    """
    Remove a URL from the knowledge base
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    url = request.get('url')
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        success = agent_instance.remove_knowledge_url(url)
        if success:
            return {
                "status": "success",
                "message": f"URL removed successfully: {url}",
                "urls": agent_instance.knowledge_urls
            }
        else:
            raise HTTPException(status_code=404, detail="URL not found")
    except Exception as e:
        logger.error(f"Error removing knowledge URL: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing URL: {str(e)}")


@app.post("/api/knowledge/reload")
async def reload_knowledge_base():
    """
    Reload the knowledge base
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Check if knowledge base can be reloaded
        if not agent_instance.openai_api_key:
            raise HTTPException(
                status_code=400, 
                detail="Cannot reload knowledge base: OpenAI API key not configured"
            )
        
        if not (Knowledge and LanceDb and OpenAIEmbedder and WebsiteReader):
            raise HTTPException(
                status_code=400,
                detail="Cannot reload knowledge base: Required modules not available. Please ensure all dependencies are installed (beautifulsoup4 may be missing)."
            )
        
        if not agent_instance.knowledge_urls:
            raise HTTPException(
                status_code=400,
                detail="Cannot reload knowledge base: No URLs configured. Please add URLs first."
            )
        
        success = agent_instance.reload_knowledge_base()
        if success:
            return {
                "status": "success",
                "message": "Knowledge base reloaded successfully"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to reload knowledge base. Check logs for details."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reloading: {str(e)}")


@app.post("/api/knowledge/clear")
async def clear_knowledge_base():
    """
    Clear all knowledge base URLs
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = agent_instance.clear_knowledge_base()
        if success:
            return {
                "status": "success",
                "message": "Knowledge base cleared successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to clear knowledge base")
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

