"""
Core Agent Implementation with RL Capabilities
Updated for Agno v2 December 2025
"""

import os
import json
import logging
import asyncio
import re
import sqlite3
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite import SqliteDb

# Import new RL components
try:
    from rl_trainer import (
        PrioritizedReplayBuffer,
        PPOTrainer,
        PPOParams,
        AdaptiveRewardWeights,
        GSPOTrainer
    )
except ImportError:
    logger.warning("RL trainer modules not available, using basic implementations")
    PrioritizedReplayBuffer = None
    PPOTrainer = None
    PPOParams = None
    AdaptiveRewardWeights = None
    GSPOTrainer = None

# Import RAG enhancement components
try:
    from rag_enhancer import EnhancedRAG
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG enhancer not available")
    EnhancedRAG = None
    ENHANCED_RAG_AVAILABLE = False

# Try to import knowledge base components - handle different agno versions
Knowledge = None
LanceDb = None
SearchType = None
OpenAIEmbedder = None
WebsiteReader = None

try:
    from agno.knowledge import Knowledge
except ImportError:
    logger.debug("Knowledge class not available")

try:
    from agno.vectordb.lancedb import LanceDb, SearchType
except ImportError:
    logger.debug("LanceDB module not available")

try:
    from agno.knowledge.embedder.openai import OpenAIEmbedder
except ImportError:
    logger.debug("OpenAI embedder module not available")

try:
    from agno.knowledge.reader.website_reader import WebsiteReader
except ImportError:
    logger.debug("Website reader module not available")


@dataclass
class RewardSignal:
    """Reward signal for RL training"""
    task_success: float
    quality_score: float
    efficiency_score: float
    user_feedback: float
    critic_score: float = 0.0
    adaptive_weights: Optional[AdaptiveRewardWeights] = None
    
    def compute_total_reward(self) -> float:
        """Compute weighted total reward"""
        if self.adaptive_weights:
            total, _ = self.adaptive_weights.compute_reward(
                self.task_success,
                self.quality_score,
                self.efficiency_score,
                self.user_feedback,
                self.critic_score
            )
            return total
        else:
            # Fallback to fixed weights
            weights = {
                'task_success': 0.35,
                'quality_score': 0.25,
                'efficiency_score': 0.1,
                'user_feedback': 0.1,
                'critic_score': 0.2
            }
            return (
                weights['task_success'] * self.task_success +
                weights['quality_score'] * self.quality_score +
                weights['efficiency_score'] * self.efficiency_score +
                weights['user_feedback'] * self.user_feedback +
                weights['critic_score'] * self.critic_score
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
        self.total_global_usage = 0
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
        """Get most relevant skills for a query using UCB-based selection"""
        logger.debug(f"Finding relevant skills for query: {query[:50]}...")
        scored_skills = []
        query_words = set(query.lower().split())
        
        # UCB Exploration Constant
        c = 0.5
        
        for skill in self.skills.values():
            desc_words = set(skill.description.lower().split())
            overlap = len(query_words.intersection(desc_words))
            
            if overlap == 0:
                continue
                
            # UCB Formula: Score = Relevance * (AverageReward + c * sqrt(ln(TotalUsage) / SkillUsage))
            if skill.usage_count > 0 and self.total_global_usage > 0:
                exploration_term = c * np.sqrt(np.log(self.total_global_usage) / skill.usage_count)
                # Cap the exploration term to avoid it dominating entirely
                exploration_term = min(exploration_term, 2.0)
                reward_term = skill.average_reward
            else:
                # High score for never-used skills to encourage exploration
                exploration_term = 1.0
                reward_term = 0.5
                
            score = overlap * (reward_term + exploration_term)
            scored_skills.append((score, skill))
        
        scored_skills.sort(reverse=True, key=lambda x: x[0])
        relevant = [skill for _, skill in scored_skills[:top_k]]
        logger.info(f"Found {len(relevant)} relevant skills using UCB: {[s.name for s in relevant]}")
        return relevant
    
    def update_skill_stats(self, name: str, reward: float, success: bool):
        """Update skill statistics after use"""
        self.total_global_usage += 1
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
        data = {
            "skills": {k: asdict(v) for k, v in self.skills.items()},
            "total_global_usage": self.total_global_usage
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_skills(self):
        """Load skills from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Support both old and new format
                    if "skills" in data:
                        skills_data = data["skills"]
                        self.total_global_usage = data.get("total_global_usage", 0)
                    else:
                        skills_data = data
                        self.total_global_usage = sum(s.get("usage_count", 0) for s in skills_data.values())
                        
                    self.skills = {k: Skill(**v) for k, v in skills_data.items()}
                logger.info(f"Loaded {len(self.skills)} skills (total usage: {self.total_global_usage}) from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Could not load skills from {self.storage_path}: {e}")
                self.skills = {}
        else:
            logger.info(f"Skills file not found at {self.storage_path}, starting with empty skill library")


class TrajectoryDatabase:
    """Persistent storage for trajectories"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize trajectory database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id TEXT UNIQUE,
                    query TEXT NOT NULL,
                    response TEXT,
                    reward REAL,
                    advantage REAL,
                    tools_used TEXT,
                    relevant_skills TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    critic_score REAL,
                    value_estimate REAL,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectory_reward 
                ON trajectories(reward DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectory_session 
                ON trajectories(session_id)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Trajectory database initialized")
        except Exception as e:
            logger.error(f"Error initializing trajectory DB: {e}")
            raise
    
    def save_trajectory(self, trajectory: Dict[str, Any]):
        """Save trajectory to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO trajectories 
                (trajectory_id, query, response, reward, advantage, tools_used, 
                 relevant_skills, session_id, user_id, critic_score, value_estimate, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.get('trajectory_id', trajectory.get('session_id', 'unknown')),
                trajectory.get('query', ''),
                trajectory.get('response', ''),
                trajectory.get('reward', 0.0),
                trajectory.get('advantage', 0.0),
                json.dumps(trajectory.get('tools_used', [])),
                json.dumps(trajectory.get('relevant_skills', [])),
                trajectory.get('session_id', ''),
                trajectory.get('user_id', ''),
                trajectory.get('critic_score', 0.0),
                trajectory.get('value_estimate', trajectory.get('reward', 0.0)),
                trajectory.get('timestamp', datetime.now().isoformat())
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")
    
    def load_trajectories(self, limit: Optional[int] = None, min_reward: Optional[float] = None) -> List[Dict[str, Any]]:
        """Load trajectories from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM trajectories"
            conditions = []
            params = []
            
            if min_reward is not None:
                conditions.append("reward >= ?")
                params.append(min_reward)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY reward DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            trajectories = []
            for row in rows:
                traj = {
                    'trajectory_id': row['trajectory_id'],
                    'query': row['query'],
                    'response': row['response'],
                    'reward': row['reward'],
                    'advantage': row['advantage'],
                    'tools_used': json.loads(row['tools_used']) if row['tools_used'] else [],
                    'relevant_skills': json.loads(row['relevant_skills']) if row['relevant_skills'] else [],
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'critic_score': row['critic_score'],
                    'value_estimate': row['value_estimate'],
                    'timestamp': row['timestamp']
                }
                trajectories.append(traj)
            
            return trajectories
        except Exception as e:
            logger.error(f"Error loading trajectories: {e}")
            return []


class TrajectoryBuffer:
    """Stores agent trajectories for RL training with prioritized replay support"""
    
    def __init__(self, max_size: int = 1000, use_prioritized: bool = True, db_path: Optional[str] = None):
        self.max_size = max_size
        self.use_prioritized = use_prioritized and PrioritizedReplayBuffer is not None
        
        if self.use_prioritized:
            self.prioritized_buffer = PrioritizedReplayBuffer(capacity=max_size)
            self.trajectories: List[Dict[str, Any]] = []  # Keep for compatibility
        else:
            self.trajectories: List[Dict[str, Any]] = []
            self.prioritized_buffer = None
        
        # Persistent storage
        self.db = None
        if db_path:
            try:
                self.db = TrajectoryDatabase(db_path)
                logger.info("Trajectory persistent storage enabled")
            except Exception as e:
                logger.warning(f"Could not initialize trajectory database: {e}")
        
        # PPO trainer for advanced advantage computation
        self.ppo_trainer = PPOTrainer() if PPOTrainer else None
    
    def add_trajectory(self, trajectory: Dict[str, Any]):
        """Add a trajectory to the buffer"""
        # Compute priority if using prioritized replay
        priority = None
        if self.use_prioritized and 'advantage' in trajectory:
            priority = abs(trajectory['advantage'])
        
        if self.use_prioritized and self.prioritized_buffer:
            self.prioritized_buffer.add(trajectory, priority)
        else:
            self.trajectories.append(trajectory)
            if len(self.trajectories) > self.max_size:
                self.trajectories.pop(0)
        
        # Save to persistent storage
        if self.db:
            try:
                self.db.save_trajectory(trajectory)
            except Exception as e:
                logger.warning(f"Could not save trajectory to DB: {e}")
    
    def get_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Get a batch of trajectories (prioritized if enabled)"""
        if self.use_prioritized and self.prioritized_buffer:
            batch, indices, weights = self.prioritized_buffer.sample(batch_size)
            # Store weights for importance sampling
            for traj, weight in zip(batch, weights):
                traj['importance_weight'] = float(weight)
            return batch
        else:
            # Fallback to random sampling
            if len(self.trajectories) < batch_size:
                return self.trajectories.copy()
            indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
            return [self.trajectories[i] for i in indices]
    
    def compute_advantages(self, trajectories: List[Dict[str, Any]], use_ppo: bool = True) -> List[float]:
        """Compute advantages using PPO-style GAE or group-relative approach"""
        if use_ppo and self.ppo_trainer:
            advantages = self.ppo_trainer.compute_advantages(trajectories, use_gae=True)
            # Store advantages in trajectories
            for traj, adv in zip(trajectories, advantages):
                traj['advantage'] = adv
            return advantages
        else:
            # Fallback to group-relative approach
            rewards = [t.get('reward', 0.0) for t in trajectories]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8
            advantages = [(r - mean_reward) / std_reward for r in rewards]
            # Store advantages
            for traj, adv in zip(trajectories, advantages):
                traj['advantage'] = adv
            return advantages
    
    def update_priorities(self, trajectories: List[Dict[str, Any]], advantages: List[float]):
        """Update priorities for trajectories (for prioritized replay)"""
        if self.use_prioritized and self.prioritized_buffer:
            # This would need trajectory indices, simplified for now
            priorities = [abs(adv) for adv in advantages]
            # In a full implementation, we'd track indices and update them
            logger.debug(f"Updated priorities for {len(priorities)} trajectories")
    
    def load_from_db(self, limit: Optional[int] = None):
        """Load trajectories from persistent storage"""
        if self.db:
            trajectories = self.db.load_trajectories(limit=limit)
            for traj in trajectories:
                self.add_trajectory(traj)
            logger.info(f"Loaded {len(trajectories)} trajectories from database")
    
    def __len__(self):
        if self.use_prioritized and self.prioritized_buffer:
            return len(self.prioritized_buffer)
        return len(self.trajectories)


class CriticAgent:
    """Evaluates agent responses to provide dense reward signals"""
    
    def __init__(self, api_key: str):
        self.agent = Agent(
            name="Critic",
            model=Claude(id="claude-3-haiku-20240307", api_key=api_key),
            instructions=[
                "You are an expert AI critic. Your job is to evaluate the quality of research assistant responses.",
                "You will be given a user query and the agent's response.",
                "Evaluate based on: Accuracy, Completeness, Relevance, and Citation quality.",
                "Output ONLY a single float number between 0.0 and 1.0 representing the score.",
                "0.0 is terrible, 1.0 is perfect."
            ],
            markdown=False
        )
        
    def evaluate(self, query: str, response: str) -> float:
        """Evaluate a response and return a score between 0.0 and 1.0"""
        if not response or not response.strip():
            return 0.0
            
        try:
            prompt = f"""
            User Query: {query}
            
            Agent Response:
            {response[:4000]} 
            
            Rate this response from 0.0 to 1.0.
            Return ONLY the number.
            """
            
            result = self.agent.run(prompt)
            content = result.content.strip()
            
            # Extract number from response
            match = re.search(r"0\.\d+|1\.0|0|1", content)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
            return 0.5 # Default if parsing fails
            
        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")
            return 0.5 # Default on error


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
        
        # Initialize adaptive reward weights
        reward_weights_path = os.path.join(os.getcwd(), "data", "reward_weights.json")
        if AdaptiveRewardWeights:
            self.adaptive_reward_weights = AdaptiveRewardWeights()
            self.adaptive_reward_weights.load(reward_weights_path)
            logger.info("Adaptive reward weights initialized")
        else:
            self.adaptive_reward_weights = None
            logger.debug("Using fixed reward weights")
        
        # Initialize trajectory buffer with persistent storage
        trajectory_db_path = os.path.join(os.path.dirname(db_path), "trajectories.db")
        self.trajectory_buffer = TrajectoryBuffer(
            max_size=10000,
            use_prioritized=True,
            db_path=trajectory_db_path
        )
        logger.debug("Trajectory buffer initialized with prioritized replay")
        
        # Load existing trajectories from database
        try:
            self.trajectory_buffer.load_from_db(limit=1000)
        except Exception as e:
            logger.warning(f"Could not load trajectories from DB: {e}")
        
        # Initialize pending trajectories cache (session_id -> trajectory)
        self.pending_trajectories: Dict[str, Dict[str, Any]] = {}

        # Initialize Critic Agent
        if self.api_key:
            self.critic = CriticAgent(api_key=self.api_key)
            logger.info("Critic Agent initialized")
        else:
            self.critic = None
            logger.warning("Critic Agent NOT initialized (missing API key)")
        
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
        self.enhanced_rag = None  # Enhanced RAG system
        self.knowledge_urls = []  # Store URLs for knowledge base
        self.lancedb_path = os.path.join(os.getcwd(), "data", "db", "lancedb")
        
        # Load saved URLs from file
        self._load_knowledge_urls()
        
        if self.openai_api_key and Knowledge and LanceDb and OpenAIEmbedder and WebsiteReader:
            try:
                logger.info("Initializing knowledge base with LanceDB...")
                if self.knowledge_urls:
                    # Create embedder
                    embedder = OpenAIEmbedder(
                        id="text-embedding-3-small",
                        api_key=self.openai_api_key
                    )
                    
                    # Create vector database
                    vector_db = LanceDb(
                        uri=self.lancedb_path,
                        table_name="research_docs",
                        search_type=SearchType.hybrid,
                        embedder=embedder
                    )
                    
                    # Create website reader
                    website_reader = WebsiteReader()
                    
                    # Create knowledge base with website reader
                    self.knowledge = Knowledge(
                        name="Research Knowledge Base",
                        description="Knowledge base for research documents",
                        vector_db=vector_db,
                        readers={"url": website_reader, "website": website_reader}
                    )
                    
                    # Load URLs into knowledge base
                    logger.info(f"Loading {len(self.knowledge_urls)} URLs into knowledge base...")
                    for url in self.knowledge_urls:
                        try:
                            self.knowledge.add_content(url=url)
                            logger.debug(f"Loaded URL: {url}")
                        except Exception as e:
                            logger.warning(f"Failed to load URL {url}: {e}")
                    
                    logger.info(f"Knowledge base initialized with {len(self.knowledge_urls)} URLs")
                    
                    # Initialize Enhanced RAG if available
                    if ENHANCED_RAG_AVAILABLE and EnhancedRAG:
                        try:
                            cohere_key = os.getenv("COHERE_API_KEY")
                            self.enhanced_rag = EnhancedRAG(
                                knowledge_base=self.knowledge,
                                use_reranking=bool(cohere_key),
                                use_multi_hop=True,
                                use_query_expansion=False,  # Can be enabled if needed
                                cohere_api_key=cohere_key
                            )
                            logger.info("Enhanced RAG system initialized")
                        except Exception as e:
                            logger.warning(f"Could not initialize Enhanced RAG: {e}")
                else:
                    # Initialize empty knowledge base
                    embedder = OpenAIEmbedder(
                        id="text-embedding-3-small",
                        api_key=self.openai_api_key
                    )
                    vector_db = LanceDb(
                        uri=self.lancedb_path,
                        table_name="research_docs",
                        search_type=SearchType.hybrid,
                        embedder=embedder
                    )
                    website_reader = WebsiteReader()
                    self.knowledge = Knowledge(
                        name="Research Knowledge Base",
                        description="Knowledge base for research documents",
                        vector_db=vector_db,
                        readers={"url": website_reader, "website": website_reader}
                    )
                    logger.info("Knowledge base initialized but no URLs configured")
                    
                    # Initialize Enhanced RAG even with empty knowledge base
                    if ENHANCED_RAG_AVAILABLE and EnhancedRAG:
                        try:
                            cohere_key = os.getenv("COHERE_API_KEY")
                            self.enhanced_rag = EnhancedRAG(
                                knowledge_base=self.knowledge,
                                use_reranking=bool(cohere_key),
                                use_multi_hop=True,
                                use_query_expansion=False,
                                cohere_api_key=cohere_key
                            )
                            logger.info("Enhanced RAG system initialized (empty knowledge base)")
                        except Exception as e:
                            logger.warning(f"Could not initialize Enhanced RAG: {e}")
            except Exception as e:
                logger.warning(f"Knowledge base not initialized: {e}", exc_info=True)
                self.knowledge = None
        else:
            missing = []
            if not self.openai_api_key:
                missing.append("OpenAI API key")
            if not Knowledge:
                missing.append("Knowledge class")
            if not LanceDb:
                missing.append("LanceDb")
            if not OpenAIEmbedder:
                missing.append("OpenAIEmbedder")
            if not WebsiteReader:
                missing.append("WebsiteReader")
            logger.info(f"Knowledge base disabled (missing: {', '.join(missing)})")
        
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
        """Add a URL to the knowledge base and sync"""
        if url not in self.knowledge_urls:
            self.knowledge_urls.append(url)
            self._save_knowledge_urls()
            
            # Load URL into knowledge base if available
            if self.knowledge:
                try:
                    logger.info(f"Loading new URL into knowledge base: {url}")
                    self.knowledge.add_content(url=url)
                    # Reload agent with new knowledge
                    self.agent = self._create_agent()
                    logger.info(f"Added URL to knowledge base and synced: {url}")
                    return True
                except Exception as e:
                    logger.error(f"Error loading URL into knowledge base: {e}", exc_info=True)
                    return False
            elif self.openai_api_key and Knowledge and LanceDb and OpenAIEmbedder and WebsiteReader:
                # Initialize knowledge base if not already initialized
                try:
                    embedder = OpenAIEmbedder(
                        id="text-embedding-3-small",
                        api_key=self.openai_api_key
                    )
                    vector_db = LanceDb(
                        uri=self.lancedb_path,
                        table_name="research_docs",
                        search_type=SearchType.hybrid,
                        embedder=embedder
                    )
                    website_reader = WebsiteReader()
                    self.knowledge = Knowledge(
                        name="Research Knowledge Base",
                        description="Knowledge base for research documents",
                        vector_db=vector_db,
                        readers={"url": website_reader, "website": website_reader}
                    )
                    # Load all URLs
                    for u in self.knowledge_urls:
                        self.knowledge.add_content(url=u)
                    self.agent = self._create_agent()
                    logger.info(f"Initialized and loaded knowledge base with {len(self.knowledge_urls)} URLs")
                    return True
                except Exception as e:
                    logger.error(f"Error initializing knowledge base: {e}", exc_info=True)
                    return False
            return True
        return False
    
    def remove_knowledge_url(self, url: str) -> bool:
        """Remove a URL from the knowledge base"""
        if url in self.knowledge_urls:
            self.knowledge_urls.remove(url)
            self._save_knowledge_urls()
            
            # Note: We can't easily remove specific URLs from LanceDB without reindexing
            # For now, we'll just update the list and reload if needed
            if self.knowledge_urls and self.knowledge:
                # Reload all remaining URLs to ensure sync
                try:
                    logger.info(f"Reloading knowledge base with {len(self.knowledge_urls)} remaining URLs (removed: {url})...")
                    # Clear and reload
                    for remaining_url in self.knowledge_urls:
                        self.knowledge.add_content(url=remaining_url)
                    self.agent = self._create_agent()
                    logger.info(f"Removed URL from knowledge base: {url}")
                    return True
                except Exception as e:
                    logger.error(f"Error reloading knowledge base: {e}", exc_info=True)
                    return True  # Still return True so URL is removed from list
            elif not self.knowledge_urls:
                # No URLs left, but keep knowledge base initialized
                logger.info("No URLs remaining in knowledge base")
                self.agent = self._create_agent()
                return True
            return True
        return False
    
    def reload_knowledge_base(self) -> bool:
        """Reload the knowledge base - reinitialize and load all URLs"""
        if not self.openai_api_key:
            logger.warning("Cannot reload knowledge base: OpenAI API key not available")
            return False
        
        if not (Knowledge and LanceDb and OpenAIEmbedder and WebsiteReader):
            logger.warning("Cannot reload knowledge base: Required modules not available")
            return False
        
        if not self.knowledge_urls:
            logger.warning("Cannot reload knowledge base: No URLs configured")
            return False
        
        try:
            logger.info(f"Reloading knowledge base with {len(self.knowledge_urls)} URLs...")
            
            # Reinitialize knowledge base to ensure sync
            embedder = OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=self.openai_api_key
            )
            vector_db = LanceDb(
                uri=self.lancedb_path,
                table_name="research_docs",
                search_type=SearchType.hybrid,
                embedder=embedder
            )
            website_reader = WebsiteReader()
            self.knowledge = Knowledge(
                name="Research Knowledge Base",
                description="Knowledge base for research documents",
                vector_db=vector_db,
                readers={"url": website_reader, "website": website_reader}
            )
            
            # Load all URLs
            for url in self.knowledge_urls:
                try:
                    self.knowledge.add_content(url=url)
                    logger.debug(f"Reloaded URL: {url}")
                except Exception as e:
                    logger.warning(f"Failed to reload URL {url}: {e}")
            
            # Reload agent
            self.agent = self._create_agent()
            logger.info("Knowledge base reloaded and synced successfully")
            return True
        except Exception as e:
            logger.error(f"Error reloading knowledge base: {e}", exc_info=True)
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
            
        # Add current datetime context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enhanced_query = f"Current Date and Time: {current_time}\n\n{enhanced_query}"
        
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
        
        # Run Critic evaluation
        critic_score = 0.0
        if self.critic:
            logger.debug("Running critic evaluation...")
            critic_score = self.critic.evaluate(query, content)
            logger.info(f"Critic score: {critic_score}")
        
        # Extract sources from response
        sources = self._extract_sources(run_response)
        
        trajectory = {
            'query': query,
            'response': content,
            'tools_used': [],
            'session_id': session_id,
            'user_id': user_id,
            'relevant_skills': [s.name for s in relevant_skills],
            'sources': sources,
            'critic_score': critic_score
        }
        
        # Cache trajectory for feedback
        if session_id:
            logger.debug(f"Caching trajectory for session {session_id}")
            self.pending_trajectories[session_id] = trajectory
        
        return {
            'response': run_response,
            'trajectory': trajectory,
            'relevant_skills': relevant_skills,
            'sources': sources,
            'critic_score': critic_score
        }
    
    async def run_task_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Execute a research task with streaming response using Agno's native async streaming
        Yields chunks of the response as they are generated
        
        Uses agent.arun(stream=True) which returns AsyncIterator[RunOutputEvent]
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
            
        # Add current datetime context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enhanced_query = f"Current Date and Time: {current_time}\n\n{enhanced_query}"
        
        try:
            accumulated_content = ""
            run_response = None
            tools_used = []
            
            logger.debug("Starting Agno native async streaming with arun()...")
            
            # Use Agno's native async streaming with arun(stream=True)
            # This returns an AsyncIterator[RunOutputEvent]
            async for event in self.agent.arun(
                enhanced_query,
                session_id=session_id,
                user_id=user_id,
                stream=True
            ):
                # Process RunOutputEvent objects
                # Each event has an 'event' attribute and possibly 'content'
                
                try:
                    event_type = getattr(event, 'event', None)
                    
                    # Handle different event types
                    if event_type == 'RunContent':
                        # This is actual content to stream
                        content = getattr(event, 'content', None)
                        
                        # Skip reasoning content (internal thinking)
                        if hasattr(event, 'reasoning_content') and event.reasoning_content:
                            continue
                        
                        if content and isinstance(content, str) and content.strip():
                            # Filter out tool execution messages
                            if 'ToolExecution' in content or 'tool_call_id' in content:
                                continue
                            
                            accumulated_content += content
                            yield {
                                'type': 'content',
                                'content': content,
                                'done': False
                            }
                    
                    elif event_type == 'ToolCallStarted':
                        # Tool is being called
                        tool_name = getattr(event, 'tool_name', 'tool')
                        
                        # Generate appropriate status message
                        if 'duckduckgo' in tool_name.lower() or 'search' in tool_name.lower():
                            status_msg = "Searching the web for information..."
                        elif 'knowledge' in tool_name.lower():
                            status_msg = "Searching knowledge base..."
                        else:
                            status_msg = f"Using {tool_name}..."
                        
                        yield {
                            'type': 'status',
                            'status': 'tool_use',
                            'message': status_msg
                        }
                    
                    elif event_type == 'ToolCallCompleted':
                        # Tool call finished
                        tool_name = getattr(event, 'tool_name', 'tool')
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                    
                    elif event_type == 'RunOutput':
                        # Final output event
                        run_response = event
                        
                        # Get final content if available
                        final_content = getattr(event, 'content', None)
                        if final_content and isinstance(final_content, str):
                            # Only add if significantly different from accumulated
                            if len(final_content) > len(accumulated_content):
                                diff = final_content[len(accumulated_content):]
                                if diff.strip():
                                    accumulated_content = final_content
                                    yield {
                                        'type': 'content',
                                        'content': diff,
                                        'done': False
                                    }
                        
                        break  # Final event, exit loop
                
                except Exception as e:
                    logger.warning(f"Error processing event: {e}")
                    continue
            
            # If we didn't get a RunOutput event, get the full response
            if not run_response:
                logger.debug("No RunOutput event received, fetching full response...")
                try:
                    run_response = await self.agent.arun(
                        enhanced_query,
                        session_id=session_id,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.warning(f"Could not get full response: {e}")

            # Run Critic evaluation (async safe wrapper)
            critic_score = 0.0
            if self.critic:
                try:
                    logger.debug("Running critic evaluation...")
                    # Run in executor to avoid blocking async loop
                    loop = asyncio.get_running_loop()
                    critic_score = await loop.run_in_executor(
                        None, 
                        lambda: self.critic.evaluate(query, accumulated_content)
                    )
                    logger.info(f"Critic score: {critic_score}")
                except Exception as e:
                    logger.error(f"Error running critic: {e}")
            
            # Extract sources and metadata from final response
            sources = []
            if run_response:
                sources = self._extract_sources(run_response)
                
                # Extract tools used from response if not already captured
                if hasattr(run_response, 'tool_calls') and run_response.tool_calls:
                    for tool_call in run_response.tool_calls:
                        tool_name = getattr(tool_call, 'name', str(tool_call))
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
            
            # Cache trajectory
            if session_id:
                trajectory = {
                    'query': query,
                    'response': accumulated_content,
                    'tools_used': tools_used,
                    'relevant_skills': [s.name for s in relevant_skills],
                    'sources': sources,
                    'critic_score': critic_score,
                    'session_id': session_id,
                    'user_id': user_id
                }
                logger.debug(f"Caching trajectory for session {session_id}")
                self.pending_trajectories[session_id] = trajectory

            # Send final done event with metadata
            yield {
                'type': 'done',
                'content': '',
                'done': True,
                'accumulated': accumulated_content,
                'tools_used': tools_used,
                'relevant_skills': [s.name for s in relevant_skills],
                'sources': sources,
                'full_response': run_response,
                'critic_score': critic_score
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
        
        # Check if we have a cached trajectory with more details (like critic score)
        session_id = trajectory.get('session_id')
        if session_id and session_id in self.pending_trajectories:
            logger.info(f"Using cached trajectory for session {session_id}")
            cached = self.pending_trajectories.pop(session_id)
            # Merge cached data into provided trajectory (prefer cached for critic_score)
            trajectory.update(cached)
            # Ensure reward_signal has critic_score from trajectory
            reward_signal.critic_score = cached.get('critic_score', 0.0)
            
        logger.info(f"Processing feedback: task_success={reward_signal.task_success}, critic_score={reward_signal.critic_score}")
        
        total_reward = reward_signal.compute_total_reward()
        trajectory['reward'] = total_reward
        trajectory['trajectory_id'] = trajectory.get('session_id', f"traj_{datetime.now().timestamp()}")
        trajectory['timestamp'] = datetime.now().isoformat()
        logger.debug(f"Total reward computed: {total_reward:.3f}")
        
        # Compute advantage for prioritized replay
        if self.trajectory_buffer.ppo_trainer:
            # Compute advantage using PPO-style GAE
            single_traj_advantages = self.trajectory_buffer.compute_advantages([trajectory], use_ppo=True)
            if single_traj_advantages:
                trajectory['advantage'] = single_traj_advantages[0]
        
        # Add to trajectory buffer (will be saved to DB if enabled)
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
            # Get recent rewards from buffer
            recent_trajectories = self.trajectory_buffer.get_batch(min(10, len(self.trajectory_buffer)))
            recent_rewards = [t.get('reward', 0.0) for t in recent_trajectories]
            if recent_rewards:
                self.training_stats['improvement_rate'] = np.mean(recent_rewards) - old_avg
        
        # Update adaptive reward weights based on performance
        if self.adaptive_reward_weights:
            component_performances = {
                'task_success': reward_signal.task_success,
                'quality_score': reward_signal.quality_score,
                'efficiency_score': reward_signal.efficiency_score,
                'user_feedback': max(0, reward_signal.user_feedback),  # Normalize to 0-1
                'critic_score': reward_signal.critic_score
            }
            self.adaptive_reward_weights.update_weights(total_reward, component_performances)
            # Save weights periodically
            if n % 50 == 0:
                reward_weights_path = os.path.join(os.getcwd(), "data", "reward_weights.json")
                self.adaptive_reward_weights.save(reward_weights_path)
        
        # Recreate agent with updated skills every 10 tasks
        if self.training_stats['total_tasks'] % 10 == 0:
            logger.info(f"Recreating agent after {self.training_stats['total_tasks']} tasks")
            self.agent = self._create_agent()
    
    def train_iteration(self, batch_size: int = 32, use_ppo: bool = True):
        """Perform a training iteration with PPO-style updates"""
        logger.info(f"Starting training iteration: batch_size={batch_size}, available_trajectories={len(self.trajectory_buffer)}")
        
        if len(self.trajectory_buffer) < batch_size:
            logger.warning(f"Not enough trajectories for training (have {len(self.trajectory_buffer)}, need {batch_size})")
            return
        
        # Get batch (prioritized if enabled)
        batch = self.trajectory_buffer.get_batch(batch_size)
        logger.debug(f"Selected batch of {len(batch)} trajectories")
        
        # Compute advantages using PPO-style GAE
        advantages = self.trajectory_buffer.compute_advantages(batch, use_ppo=use_ppo)
        logger.debug(f"Computed advantages: mean={np.mean(advantages):.3f}, std={np.std(advantages):.3f}")
        
        # Update priorities for prioritized replay
        if self.trajectory_buffer.use_prioritized:
            self.trajectory_buffer.update_priorities(batch, advantages)
        
        # Update skill weights based on advantages
        updated_skills = 0
        for traj, advantage in zip(batch, advantages):
            for skill_name in traj.get('relevant_skills', []):
                if skill_name in self.skill_library.skills:
                    skill = self.skill_library.skills[skill_name]
                    if advantage > 0:
                        # PPO-style soft update
                        boost_factor = 1.0 + min(0.1, advantage * 0.05)  # Cap boost
                        old_rate = skill.success_rate
                        skill.success_rate = min(1.0, skill.success_rate * boost_factor)
                        logger.debug(f"Boosted skill {skill_name}: {old_rate:.3f} -> {skill.success_rate:.3f} (advantage={advantage:.3f})")
                        updated_skills += 1
                    elif advantage < -0.5:  # Significant negative advantage
                        # Slight decay for poor performance
                        old_rate = skill.success_rate
                        skill.success_rate = max(0.0, skill.success_rate * 0.98)
                        logger.debug(f"Decayed skill {skill_name}: {old_rate:.3f} -> {skill.success_rate:.3f}")
        
        self.skill_library.save_skills()
        logger.info(f"Training iteration completed. Batch size: {batch_size}, skills updated: {updated_skills}")
    
    def _extract_sources(self, run_response) -> List[Dict[str, str]]:
        """Extract sources from agent response"""
        sources = []
        
        try:
            # Extract from knowledge base if used
            if hasattr(run_response, 'knowledge_sources') and run_response.knowledge_sources:
                for kb_source in run_response.knowledge_sources:
                    sources.append({
                        'type': 'rag',
                        'identifier': 'RAG',
                        'url': getattr(kb_source, 'url', ''),
                        'title': getattr(kb_source, 'title', ''),
                        'source': 'Knowledge Base'
                    })
            
            # Extract from tool calls (web search URLs)
            if hasattr(run_response, 'tool_calls') and run_response.tool_calls:
                for tool_call in run_response.tool_calls:
                    tool_name = getattr(tool_call, 'name', str(tool_call)).lower()
                    if 'duckduckgo' in tool_name or 'search' in tool_name:
                        # Extract URLs from web search results
                        if hasattr(tool_call, 'result') and tool_call.result:
                            if isinstance(tool_call.result, list):
                                for result in tool_call.result:
                                    if isinstance(result, dict):
                                        url = result.get('url') or result.get('link') or result.get('href')
                                        title = result.get('title') or result.get('name')
                                        if url:
                                            sources.append({
                                                'type': 'web_search',
                                                'identifier': 'Web',
                                                'url': url,
                                                'title': title or url,
                                                'source': 'DuckDuckGo Search'
                                            })
                            elif isinstance(tool_call.result, dict):
                                url = tool_call.result.get('url') or tool_call.result.get('link')
                                title = tool_call.result.get('title')
                                if url:
                                    sources.append({
                                        'type': 'web_search',
                                        'identifier': 'Web',
                                        'url': url,
                                        'title': title or url,
                                        'source': 'DuckDuckGo Search'
                                    })
            
            # Extract URLs from knowledge base URLs if RAG was used
            if self.knowledge and self.knowledge_urls:
                # Check if knowledge was actually used in the response
                # This is a heuristic - in a real implementation, Agno would track this
                for kb_url in self.knowledge_urls:
                    if kb_url not in [s.get('url', '') for s in sources]:
                        sources.append({
                            'type': 'rag',
                            'identifier': 'RAG',
                            'url': kb_url,
                            'title': kb_url,
                            'source': 'Knowledge Base'
                        })
            
            # Remove duplicates
            seen = set()
            unique_sources = []
            for source in sources:
                key = source['url']
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(source)
            
            logger.debug(f"Extracted {len(unique_sources)} sources from response")
            return unique_sources
            
        except Exception as e:
            logger.warning(f"Error extracting sources: {e}")
            return []
    
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

