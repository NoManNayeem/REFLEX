"""
Core Agent Implementation with RL Capabilities
Updated for Agno v2 December 2025
"""

import os
import json
import logging
import re
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
                            self.knowledge.load(url=url, upsert=True)
                            logger.debug(f"Loaded URL: {url}")
                        except Exception as e:
                            logger.warning(f"Failed to load URL {url}: {e}")
                    
                    logger.info(f"Knowledge base initialized with {len(self.knowledge_urls)} URLs")
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
                    self.knowledge.load(url=url, upsert=True)
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
                        self.knowledge.load(url=u, upsert=True)
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
                        self.knowledge.load(url=remaining_url, upsert=True)
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
                    self.knowledge.load(url=url, upsert=True)
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
        
        # Extract sources from response
        sources = self._extract_sources(run_response)
        
        trajectory = {
            'query': query,
            'response': content,
            'tools_used': [],
            'session_id': session_id,
            'user_id': user_id,
            'relevant_skills': [s.name for s in relevant_skills],
            'sources': sources
        }
        
        return {
            'response': run_response,
            'trajectory': trajectory,
            'relevant_skills': relevant_skills,
            'sources': sources
        }
    
    async def run_task_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Execute a research task with streaming response using Agno's native streaming
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
        
        # Use Agno's native streaming
        try:
            import asyncio
            accumulated_content = ""
            run_response = None
            tools_used = []
            
            # Use agent.run with stream=True to get real-time streaming
            logger.debug("Starting Agno agent streaming...")
            
            # Run agent with streaming enabled
            loop = asyncio.get_event_loop()
            
            # Check if agent.run supports streaming
            # Agno's run() with stream=True returns an iterator/generator
            stream_result = await loop.run_in_executor(
                None,
                lambda: self.agent.run(
                    enhanced_query,
                    session_id=session_id,
                    user_id=user_id,
                    stream=True
                )
            )
            
            # Process streaming chunks from Agno
            if hasattr(stream_result, '__iter__'):
                logger.debug("Processing streaming chunks from Agno...")
                
                # Convert sync iterator to async
                def sync_iter():
                    try:
                        for chunk in stream_result:
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error in sync iterator: {e}", exc_info=True)
                
                # Process chunks - Agno yields RunContentEvent objects
                last_tool_status = None
                for chunk in sync_iter():
                    # Check for tool calls or tool-related events
                    if hasattr(chunk, 'event'):
                        event_type = chunk.event
                        
                        # Detect tool calls - check for tools attribute or tool-related events
                        if hasattr(chunk, 'tools') and chunk.tools:
                            # Tool is being called
                            tool_names = []
                            if isinstance(chunk.tools, list):
                                tool_names = [str(t) for t in chunk.tools]
                            elif hasattr(chunk.tools, '__iter__'):
                                tool_names = [str(t) for t in chunk.tools]
                            
                            # Generate status message for tool usage
                            for tool_name in tool_names:
                                if 'duckduckgo' in tool_name.lower() or 'search' in tool_name.lower():
                                    status_msg = "Searching the web for information..."
                                    if status_msg != last_tool_status:
                                        yield {
                                            'type': 'status',
                                            'status': 'searching',
                                            'message': status_msg
                                        }
                                        last_tool_status = status_msg
                                elif 'knowledge' in tool_name.lower() or 'rag' in tool_name.lower():
                                    status_msg = "Searching knowledge base..."
                                    if status_msg != last_tool_status:
                                        yield {
                                            'type': 'status',
                                            'status': 'searching',
                                            'message': status_msg
                                        }
                                        last_tool_status = status_msg
                                else:
                                    status_msg = f"Using {tool_name}..."
                                    if status_msg != last_tool_status:
                                        yield {
                                            'type': 'status',
                                            'status': 'tool_use',
                                            'message': status_msg
                                        }
                                        last_tool_status = status_msg
                    
                    # Handle content chunks
                    content_to_yield = None
                    
                    # Handle RunContentEvent objects (Agno's streaming format)
                    if hasattr(chunk, 'content') and hasattr(chunk, 'event'):
                        # This is a RunContentEvent
                        if chunk.event == 'RunContent' or chunk.event == 'content':
                            content = chunk.content
                            
                            # Skip reasoning content - it's internal thinking
                            if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                                # This is reasoning, not the final response - skip it
                                continue
                            
                            if content:
                                # Content might be string or other type
                                if isinstance(content, str):
                                    content_to_yield = content
                                else:
                                    # Try to convert to string
                                    content_to_yield = str(content)
                    elif hasattr(chunk, 'content') and not hasattr(chunk, 'event'):
                        # Direct content attribute (might be RunOutput)
                        content = chunk.content
                        if isinstance(content, str) and content:
                            content_to_yield = content
                        else:
                            # This might be the final RunOutput
                            run_response = chunk
                    elif hasattr(chunk, 'text'):
                        # Text attribute
                        content_to_yield = chunk.text if isinstance(chunk.text, str) else str(chunk.text) if chunk.text else None
                    elif isinstance(chunk, str):
                        # Direct string
                        content_to_yield = chunk
                    
                    # Yield content if we have it (filter out status-like messages and internal details)
                    if content_to_yield:
                        # Filter out common status messages that should be shown as status, not content
                        status_patterns = [
                            r"^I'll search.*",
                            r"^Let me search.*",
                            r"^Let me try.*",
                            r"^Searching for.*",
                            r"^I'm searching.*",
                            r"^details about.*:.*search.*",
                            r"^.*: search.*",
                        ]
                        is_status_message = any(re.match(pattern, content_to_yield.strip(), re.IGNORECASE) for pattern in status_patterns)
                        
                        # Filter out tool execution details and internal reasoning
                        if 'ToolExecution' in content_to_yield or 'tool_call_id' in content_to_yield:
                            # This is tool execution metadata - skip it
                            continue
                        
                        # Filter out patterns that look like internal reasoning
                        # Match patterns like "details about X: search..." or "X: search..."
                        if re.search(r'^[^:]+:\s*(search|find|look|get)', content_to_yield.strip(), re.IGNORECASE):
                            # Looks like internal reasoning format "topic: search..."
                            logger.debug(f"Filtered out internal reasoning: {content_to_yield[:50]}...")
                            continue
                        
                        # Filter out content that starts with reasoning-like patterns
                        if re.match(r'^(details about|information about|let me|I will|I\'ll).*:\s*(search|find)', content_to_yield.strip(), re.IGNORECASE):
                            logger.debug(f"Filtered out reasoning pattern: {content_to_yield[:50]}...")
                            continue
                        
                        if not is_status_message:
                            accumulated_content += content_to_yield
                            yield {
                                'type': 'content',
                                'content': content_to_yield,
                                'done': False
                            }
                    
                    # Check if this is the final chunk (RunOutput)
                    if hasattr(chunk, '__class__'):
                        class_name = str(chunk.__class__)
                        if 'RunOutput' in class_name and not hasattr(chunk, 'event'):
                            # Final RunOutput object
                            run_response = chunk
                            break
                
                # If we didn't get a final response object, get it from the last chunk
                if not run_response:
                    # Try to get the full response
                    try:
                        # Run again without streaming to get full response for metadata
                        run_response = await loop.run_in_executor(
                            None,
                            lambda: self.agent.run(
                                enhanced_query,
                                session_id=session_id,
                                user_id=user_id
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Could not get full response for metadata: {e}")
                
            else:
                # Fallback: if stream=True doesn't return iterator, use word-by-word chunking
                logger.debug("Stream result is not iterable, using fallback chunking...")
                run_response = stream_result
                content = run_response.content if hasattr(run_response, 'content') and run_response.content else str(stream_result)
                
                # Stream word by word for better UX
                words = content.split()
                accumulated_content = ""
                
                for word in words:
                    accumulated_content += word + " "
                    yield {
                        'type': 'content',
                        'content': word + " ",
                        'done': False
                    }
                    await asyncio.sleep(0.02)
            
            # Extract sources and tools from final response
            if run_response:
                sources = self._extract_sources(run_response)
                
                # Extract tools used
                if hasattr(run_response, 'tool_calls') and run_response.tool_calls:
                    tools_used = [tool.name if hasattr(tool, 'name') else str(tool) for tool in run_response.tool_calls]
                elif hasattr(run_response, 'tools') and run_response.tools:
                    if isinstance(run_response.tools, list):
                        tools_used = [str(t) for t in run_response.tools]
                    else:
                        tools_used = [str(run_response.tools)]
            else:
                sources = []
                tools_used = []
            
            # Final chunk with metadata
            yield {
                'type': 'done',
                'content': '',
                'done': True,
                'accumulated': accumulated_content,
                'tools_used': tools_used,
                'relevant_skills': [s.name for s in relevant_skills],
                'sources': sources,
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

