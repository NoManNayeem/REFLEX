"""
Pydantic Models for API Request/Response
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    trajectory_id: str = Field(..., description="Trajectory ID for feedback")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in response")
    relevant_skills: List[str] = Field(default_factory=list, description="Skills applied")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Sources used (RAG, web search, etc.)")
    critic_score: float = Field(0.0, description="Internal critic score")
    timestamp: str = Field(..., description="Response timestamp")


class SkillBase(BaseModel):
    """Base model for skill"""
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    context: str = Field(..., description="Context/approach for using the skill")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")


class SkillCreate(SkillBase):
    """Model for creating a new skill"""
    pass


class SkillResponse(SkillBase):
    """Response model for skill"""
    usage_count: int = Field(..., description="Number of times skill was used")
    average_reward: float = Field(..., description="Average reward received")


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint"""
    session_id: str = Field(..., description="Session ID")
    task_success: float = Field(..., ge=0.0, le=1.0, description="Task success score (0-1)")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score (0-1)")
    efficiency_score: float = Field(..., ge=0.0, le=1.0, description="Efficiency score (0-1)")
    user_feedback: float = Field(..., ge=-1.0, le=1.0, description="User feedback (-1 to 1)")
    learned_skill: Optional[SkillCreate] = Field(None, description="Optional skill learned from task")


class TrainingRequest(BaseModel):
    """Request model for training endpoint"""
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for training")


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    user_id: str
    message_count: int
    last_activity: str


class AgentStats(BaseModel):
    """Agent statistics"""
    total_tasks: int = Field(..., description="Total tasks completed")
    successful_tasks: int = Field(..., description="Successfully completed tasks")
    average_reward: float = Field(..., description="Average reward across all tasks")
    improvement_rate: float = Field(..., description="Rate of improvement")
    skill_count: int = Field(..., description="Number of learned skills")
    trajectory_count: int = Field(..., description="Number of stored trajectories")
    top_skills: List[Dict[str, Any]] = Field(default_factory=list, description="Top performing skills")

