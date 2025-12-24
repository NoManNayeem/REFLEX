"""
Core Agent Implementation with RL Capabilities
Updated for Agno v2 December 2025
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb

# Try to import knowledge base components - handle different agno versions
WebsiteKnowledgeBase = None
LanceDb = None
SearchType = None
OpenAIEmbedder = None

try:
    from agno.knowledge.website import WebsiteKnowledgeBase
except ImportError:
    try:
        from agno.knowledge.url import UrlKnowledge as WebsiteKnowledgeBase
    except ImportError:
        logger.debug("Knowledge base module not available")

try:
    from agno.vectordb.lancedb import LanceDb, SearchType
except ImportError:
    logger.debug("LanceDB module not available")

try:
    from agno.embedder.openai import OpenAIEmbedder
except ImportError:
    logger.debug("OpenAI embedder module not available")


@dataclass
class RewardSignal:
    """Reward signal for RL training"""
    task_success: float
    quality_score: float
    efficiency_score: float
    user_feedback: float
    
    def compute_total_reward(self) -> float:
        """Compute weighted total reward"""
        weights = {
            'task_success': 0.4,
            'quality_score': 0.3,
            'efficiency_score': 0.15,
            'user_feedback': 0.15
        }
        return (
            weights['task_success'] * self.task_success +
            weights['quality_score'] * self.quality_score +
            weights['efficiency_score'] * self.efficiency_score +
            weights['user_feedback'] * self.user_feedback
        )


@dataclass
class Skill:
    """A learned skill that can be reused"""
    name: str
    description: str
    context: str
    success_rate: float
    usage_count: int
    average_reward: float


class SkillLibrary:
    """Manages learned skills for self-improvement"""
    
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Use absolute path to ensure it works in Docker
            storage_path = os.path.join(os.getcwd(), "data", "skills", "skills.json")
        self.storage_path = storage_path
        self.skills: Dict[str, Skill] = {}
        self.load_skills()
    
    def add_skill(self, skill: Skill):
        """Add or update a skill"""
        logger.info(f"Adding/updating skill: {skill.name} (success_rate={skill.success_rate:.2f})")
        self.skills[skill.name] = skill
        self.save_skills()
        logger.debug(f"Skill library now has {len(self.skills)} skills")
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve a skill by name"""
        return self.skills.get(name)
    
    def get_relevant_skills(self, query: str, top_k: int = 3) -> List[Skill]:
        """Get most relevant skills for a query"""
        logger.debug(f"Finding relevant skills for query: {query[:50]}...")
        scored_skills = []
        query_words = set(query.lower().split())
        
        for skill in self.skills.values():
            desc_words = set(skill.description.lower().split())
            overlap = len(query_words.intersection(desc_words))
            score = overlap * skill.success_rate * (1 + np.log1p(skill.usage_count))
            scored_skills.append((score, skill))
        
        scored_skills.sort(reverse=True, key=lambda x: x[0])
        relevant = [skill for _, skill in scored_skills[:top_k]]
        logger.info(f"Found {len(relevant)} relevant skills: {[s.name for s in relevant]}")
        return relevant
    
    def update_skill_stats(self, name: str, reward: float, success: bool):
        """Update skill statistics after use"""
        if name in self.skills:
            skill = self.skills[name]
            old_success_rate = skill.success_rate
            skill.usage_count += 1
            skill.average_reward = (
                (skill.average_reward * (skill.usage_count - 1) + reward) / 
                skill.usage_count
            )
            if success:
                skill.success_rate = (
                    (skill.success_rate * (skill.usage_count - 1) + 1.0) / 
                    skill.usage_count
                )
            else:
                skill.success_rate = (
                    (skill.success_rate * (skill.usage_count - 1)) / 
                    skill.usage_count
                )
            logger.debug(f"Updated skill {name}: usage={skill.usage_count}, success_rate={old_success_rate:.2f}->{skill.success_rate:.2f}, reward={reward:.2f}")
            self.save_skills()
        else:
            logger.warning(f"Attempted to update non-existent skill: {name}")
    
    def save_skills(self):
        """Persist skills to disk"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.skills.items()}, f, indent=2)
    
    def load_skills(self):
        """Load skills from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.skills = {k: Skill(**v) for k, v in data.items()}
                logger.info(f"Loaded {len(self.skills)} skills from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Could not load skills from {self.storage_path}: {e}")
                self.skills = {}
        else:
            logger.info(f"Skills file not found at {self.storage_path}, starting with empty skill library")


class TrajectoryBuffer:
    """Stores agent trajectories for RL training"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.trajectories: List[Dict[str, Any]] = []
    
    def add_trajectory(self, trajectory: Dict[str, Any]):
        """Add a trajectory to the buffer"""
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.max_size:
            self.trajectories.pop(0)
    
    def get_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get a random batch of trajectories"""
        if len(self.trajectories) < batch_size:
            return self.trajectories.copy()
        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        return [self.trajectories[i] for i in indices]
    
    def compute_advantages(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """Compute advantages using group-relative approach"""
        rewards = [t['reward'] for t in trajectories]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages


class SelfImprovingResearchAgent:
    """Self-improving research agent with RL capabilities"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        db_path: str = None
    ):
        if db_path is None:
            # Use absolute path to ensure it works in Docker
            db_path = os.path.join(os.getcwd(), "data", "db", "agent.db")
        logger.info("Initializing SelfImprovingResearchAgent...")
        # Load .env from root if not already loaded
        from dotenv import load_dotenv
        import pathlib
        root_env = pathlib.Path(__file__).parent.parent / ".env"
        if root_env.exists():
            load_dotenv(root_env)
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        logger.debug(f"API keys loaded: Anthropic={bool(self.api_key)}, OpenAI={bool(self.openai_api_key)}")
        
        # Initialize skill library
        self.skill_library = SkillLibrary()
        logger.debug(f"Skill library initialized with {len(self.skill_library.skills)} skills")
        
        # Initialize trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer()
        logger.debug("Trajectory buffer initialized")
        
        # Training statistics
        self.training_stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'average_reward': 0.0,
            'improvement_rate': 0.0
        }
        
        # Setup database for memory and storage
        self.db = SqliteDb(db_file=db_path)
        
        # Setup knowledge base (optional - can be None for POC)
        self.knowledge = None
        self.knowledge_urls = []  # Store URLs for knowledge base
        self.lancedb_path = os.path.join(os.getcwd(), "data", "db", "lancedb")
        
        # Load saved URLs from file
        self._load_knowledge_urls()
        
        if self.openai_api_key and WebsiteKnowledgeBase and LanceDb and OpenAIEmbedder:
            try:
                logger.info("Initializing knowledge base with LanceDB...")
                if self.knowledge_urls:
                    self.knowledge = WebsiteKnowledgeBase(
                        urls=self.knowledge_urls,
                        vector_db=LanceDb(
                            uri=self.lancedb_path,
                            table_name="research_docs",
                            search_type=SearchType.hybrid,
                            embedder=OpenAIEmbedder(
                                id="text-embedding-3-small",
                                dimensions=1536,
                                api_key=self.openai_api_key
                            )
                        )
                    )
                    logger.info(f"Knowledge base initialized with {len(self.knowledge_urls)} URLs")
                else:
                    logger.info("Knowledge base initialized but no URLs configured")
            except Exception as e:
                logger.warning(f"Knowledge base not initialized: {e}")
                self.knowledge = None
        else:
            if not self.openai_api_key:
                logger.info("Knowledge base disabled (no OpenAI API key)")
            else:
                logger.info("Knowledge base disabled (required modules not available)")
        
        # Create the main agent
        self.agent = self._create_agent()
    
    def _load_knowledge_urls(self):
        """Load knowledge base URLs from file"""
        urls_file = os.path.join(os.getcwd(), "data", "knowledge_urls.json")
        try:
            if os.path.exists(urls_file):
                with open(urls_file, 'r') as f:
                    data = json.load(f)
                    self.knowledge_urls = data.get('urls', [])
                    logger.info(f"Loaded {len(self.knowledge_urls)} knowledge base URLs")
            else:
                # Default URL
                self.knowledge_urls = ["https://docs.agno.com/"]
                self._save_knowledge_urls()
        except Exception as e:
            logger.warning(f"Error loading knowledge URLs: {e}")
            self.knowledge_urls = []
    
    def _save_knowledge_urls(self):
        """Save knowledge base URLs to file"""
        urls_file = os.path.join(os.getcwd(), "data", "knowledge_urls.json")
        try:
            os.makedirs(os.path.dirname(urls_file), exist_ok=True)
            with open(urls_file, 'w') as f:
                json.dump({'urls': self.knowledge_urls}, f, indent=2)
            logger.debug(f"Saved {len(self.knowledge_urls)} knowledge base URLs")
        except Exception as e:
            logger.error(f"Error saving knowledge URLs: {e}")
    
    def add_knowledge_url(self, url: str) -> bool:
        """Add a URL to the knowledge base"""
        if url not in self.knowledge_urls:
            self.knowledge_urls.append(url)
            self._save_knowledge_urls()
            
            # Reinitialize knowledge base if available
            if self.openai_api_key and WebsiteKnowledgeBase and LanceDb and OpenAIEmbedder:
                try:
                    self.knowledge = WebsiteKnowledgeBase(
                        urls=self.knowledge_urls,
                        vector_db=LanceDb(
                            uri=self.lancedb_path,
                            table_name="research_docs",
                            search_type=SearchType.hybrid,
                            embedder=OpenAIEmbedder(
                                id="text-embedding-3-small",
                                dimensions=1536,
                                api_key=self.openai_api_key
                            )
                        )
                    )
                    # Reload agent with new knowledge
                    self.agent = self._create_agent()
                    logger.info(f"Added URL to knowledge base: {url}")
                    return True
                except Exception as e:
                    logger.error(f"Error reinitializing knowledge base: {e}")
                    return False
            return True
        return False
    
    def remove_knowledge_url(self, url: str) -> bool:
        """Remove a URL from the knowledge base"""
        if url in self.knowledge_urls:
            self.knowledge_urls.remove(url)
            self._save_knowledge_urls()
            
            # Reinitialize knowledge base
            if self.openai_api_key and WebsiteKnowledgeBase and LanceDb and OpenAIEmbedder:
                try:
                    if self.knowledge_urls:
                        self.knowledge = WebsiteKnowledgeBase(
                            urls=self.knowledge_urls,
                            vector_db=LanceDb(
                                uri=self.lancedb_path,
                                table_name="research_docs",
                                search_type=SearchType.hybrid,
                                embedder=OpenAIEmbedder(
                                    id="text-embedding-3-small",
                                    dimensions=1536,
                                    api_key=self.openai_api_key
                                )
                            )
                        )
                    else:
                        self.knowledge = None
                    # Reload agent
                    self.agent = self._create_agent()
                    logger.info(f"Removed URL from knowledge base: {url}")
                    return True
                except Exception as e:
                    logger.error(f"Error reinitializing knowledge base: {e}")
                    return False
            return True
        return False
    
    def reload_knowledge_base(self) -> bool:
        """Reload the knowledge base"""
        if self.knowledge and self.knowledge_urls:
            try:
                logger.info("Reloading knowledge base...")
                self.knowledge.load(upsert=True)
                logger.info("Knowledge base reloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error reloading knowledge base: {e}")
                return False
        return False
    
    def clear_knowledge_base(self) -> bool:
        """Clear all knowledge base URLs"""
        self.knowledge_urls = []
        self._save_knowledge_urls()
        self.knowledge = None
        self.agent = self._create_agent()
        logger.info("Knowledge base cleared")
        return True
    
    def _create_agent(self) -> Agent:
        """Create the research agent with all capabilities"""
        
        base_instructions = [
            "You are a research assistant with self-improvement capabilities.",
            "Use web search to find current information when needed.",
            "Be thorough in your research and provide well-sourced answers.",
            "Learn from feedback to improve your performance over time."
        ]
        
        # Add skill context
        if self.skill_library.skills:
            skill_context = "\n\nLearned Skills:\n"
            for skill in list(self.skill_library.skills.values())[:5]:
                skill_context += (
                    f"- {skill.name}: {skill.description} "
                    f"(success rate: {skill.success_rate:.2f})\n"
                )
            base_instructions.append(skill_context)
        
        agent_config = {
            "name": "Research Agent",
            "model": Claude(id="claude-sonnet-4-20250514", api_key=self.api_key),
            "instructions": base_instructions,
            "tools": [DuckDuckGoTools()],
            "db": self.db,
            "add_history_to_context": True,
            "markdown": True
        }
        
        # Add knowledge if available
        if self.knowledge:
            agent_config["knowledge"] = self.knowledge
            agent_config["search_knowledge"] = True
        
        logger.debug(f"Creating agent with config keys: {list(agent_config.keys())}")
        return Agent(**agent_config)
    
    def run_task(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a research task"""
        logger.info(f"Running task: session_id={session_id}, query_length={len(query)}")
        
        # Get relevant skills
        relevant_skills = self.skill_library.get_relevant_skills(query)
        logger.debug(f"Retrieved {len(relevant_skills)} relevant skills")
        
        # Add skill context if relevant
        if relevant_skills:
            skill_text = "\n\nRelevant learned approaches:\n"
            for skill in relevant_skills:
                skill_text += f"- {skill.name}: {skill.context}\n"
            enhanced_query = query + skill_text
        else:
            enhanced_query = query
        
        # Run the agent
        logger.debug("Calling agent.run()...")
        run_response = self.agent.run(
            enhanced_query,
            session_id=session_id,
            user_id=user_id
        )
        
        # Extract content
        content = run_response.content if run_response.content else ""
        logger.info(f"Agent response received: length={len(content)}")
        
        trajectory = {
            'query': query,
            'response': content,
            'tools_used': [],
            'session_id': session_id,
            'user_id': user_id,
            'relevant_skills': [s.name for s in relevant_skills]
        }
        
        return {
            'response': run_response,
            'trajectory': trajectory,
            'relevant_skills': relevant_skills
        }
    
    async def run_task_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Execute a research task with streaming response
        Yields chunks of the response as they are generated
        """
        logger.info(f"Running streaming task: session_id={session_id}, query_length={len(query)}")
        
        # Get relevant skills
        relevant_skills = self.skill_library.get_relevant_skills(query)
        logger.debug(f"Retrieved {len(relevant_skills)} relevant skills")
        
        # Add skill context if relevant
        if relevant_skills:
            skill_text = "\n\nRelevant learned approaches:\n"
            for skill in relevant_skills:
                skill_text += f"- {skill.name}: {skill.context}\n"
            enhanced_query = query + skill_text
        else:
            enhanced_query = query
        
        # Prepare trajectory info
        trajectory = {
            'query': query,
            'response': '',
            'tools_used': [],
            'session_id': session_id,
            'user_id': user_id,
            'relevant_skills': [s.name for s in relevant_skills]
        }
        
        # Try to use model's streaming capability
        try:
            # Check if agent's model supports streaming
            model = self.agent.model
            if hasattr(model, 'stream') or hasattr(model, 'stream_response'):
                logger.debug("Using model streaming capability...")
                
                # Use agent's run with streaming
                # Agno's Agent.run() may support streaming through the model
                # We'll need to check the actual implementation
                # For now, we'll use a workaround: run the agent and stream the response
                
                # Run agent (this might block, but we'll handle it)
                import asyncio
                loop = asyncio.get_event_loop()
                run_response = await loop.run_in_executor(
                    None,
                    lambda: self.agent.run(
                        enhanced_query,
                        session_id=session_id,
                        user_id=user_id
                    )
                )
                
                # Stream the response content
                content = run_response.content if run_response.content else ""
                logger.info(f"Agent response received: length={len(content)}, streaming chunks...")
                
                # Stream content in chunks (simulate real streaming by chunking)
                # In a real implementation, this would come from the model's stream
                chunk_size = 20  # Characters per chunk
                accumulated_content = ""
                
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    accumulated_content += chunk
                    
                    yield {
                        'type': 'content',
                        'content': chunk,
                        'done': False,
                        'accumulated': accumulated_content
                    }
                    
                    # Small delay to simulate real streaming
                    await asyncio.sleep(0.02)
                
                # Final chunk with metadata
                yield {
                    'type': 'done',
                    'content': '',
                    'done': True,
                    'accumulated': accumulated_content,
                    'tools_used': [],
                    'relevant_skills': [s.name for s in relevant_skills],
                    'full_response': run_response
                }
                
            else:
                # Fallback: run normally and stream the result
                logger.debug("Model doesn't support streaming, using fallback...")
                import asyncio
                loop = asyncio.get_event_loop()
                run_response = await loop.run_in_executor(
                    None,
                    lambda: self.agent.run(
                        enhanced_query,
                        session_id=session_id,
                        user_id=user_id
                    )
                )
                
                content = run_response.content if run_response.content else ""
                
                # Stream word by word for better UX
                words = content.split()
                accumulated_content = ""
                
                for i, word in enumerate(words):
                    accumulated_content += word + " "
                    yield {
                        'type': 'content',
                        'content': word + " ",
                        'done': False,
                        'accumulated': accumulated_content
                    }
                    await asyncio.sleep(0.03)
                
                # Final yield with metadata
                yield {
                    'type': 'done',
                    'content': '',
                    'done': True,
                    'accumulated': accumulated_content,
                    'tools_used': [],
                    'relevant_skills': [s.name for s in relevant_skills],
                    'full_response': run_response
                }
                
        except Exception as e:
            logger.error(f"Error in streaming task: {str(e)}", exc_info=True)
            yield {
                'type': 'error',
                'content': f'Error: {str(e)}',
                'done': True,
                'error': str(e)
            }
    
    def provide_feedback(
        self,
        trajectory: Dict[str, Any],
        reward_signal: RewardSignal,
        learned_skill: Optional[Skill] = None
    ):
        """Process feedback and update the agent"""
        logger.info(f"Processing feedback: task_success={reward_signal.task_success}, quality={reward_signal.quality_score}")
        
        total_reward = reward_signal.compute_total_reward()
        trajectory['reward'] = total_reward
        logger.debug(f"Total reward computed: {total_reward:.3f}")
        
        # Add to trajectory buffer
        self.trajectory_buffer.add_trajectory(trajectory)
        
        # Update skill library
        if learned_skill:
            self.skill_library.add_skill(learned_skill)
        
        # Update statistics for used skills
        for skill_name in trajectory.get('relevant_skills', []):
            success = reward_signal.task_success > 0.7
            self.skill_library.update_skill_stats(skill_name, total_reward, success)
        
        # Update training statistics
        self.training_stats['total_tasks'] += 1
        if reward_signal.task_success > 0.7:
            self.training_stats['successful_tasks'] += 1
        
        old_avg = self.training_stats['average_reward']
        n = self.training_stats['total_tasks']
        self.training_stats['average_reward'] = (old_avg * (n - 1) + total_reward) / n
        
        # Compute improvement rate
        if n > 10:
            recent_rewards = [
                t['reward'] for t in self.trajectory_buffer.trajectories[-10:]
            ]
            self.training_stats['improvement_rate'] = np.mean(recent_rewards) - old_avg
        
        # Recreate agent with updated skills every 10 tasks
        if self.training_stats['total_tasks'] % 10 == 0:
            logger.info(f"Recreating agent after {self.training_stats['total_tasks']} tasks")
            self.agent = self._create_agent()
    
    def train_iteration(self, batch_size: int = 32):
        """Perform a training iteration"""
        logger.info(f"Starting training iteration: batch_size={batch_size}, available_trajectories={len(self.trajectory_buffer.trajectories)}")
        
        if len(self.trajectory_buffer.trajectories) < batch_size:
            logger.warning(f"Not enough trajectories for training (have {len(self.trajectory_buffer.trajectories)}, need {batch_size})")
            return
        
        # Get batch
        batch = self.trajectory_buffer.get_batch(batch_size)
        logger.debug(f"Selected batch of {len(batch)} trajectories")
        
        # Compute advantages
        advantages = self.trajectory_buffer.compute_advantages(batch)
        logger.debug(f"Computed advantages: mean={np.mean(advantages):.3f}, std={np.std(advantages):.3f}")
        
        # Update skill weights based on advantages
        updated_skills = 0
        for traj, advantage in zip(batch, advantages):
            for skill_name in traj.get('relevant_skills', []):
                if skill_name in self.skill_library.skills:
                    skill = self.skill_library.skills[skill_name]
                    if advantage > 0:
                        old_rate = skill.success_rate
                        skill.success_rate = min(1.0, skill.success_rate * 1.05)
                        logger.debug(f"Boosted skill {skill_name}: {old_rate:.3f} -> {skill.success_rate:.3f}")
                        updated_skills += 1
        
        self.skill_library.save_skills()
        logger.info(f"Training iteration completed. Batch size: {batch_size}, skills updated: {updated_skills}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_stats': self.training_stats,
            'skill_count': len(self.skill_library.skills),
            'trajectory_count': len(self.trajectory_buffer.trajectories),
            'top_skills': [
                {'name': s.name, 'success_rate': s.success_rate, 'usage': s.usage_count}
                for s in sorted(
                    self.skill_library.skills.values(),
                    key=lambda x: x.success_rate * x.usage_count,
                    reverse=True
                )[:5]
            ]
        }

