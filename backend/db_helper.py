"""
Database helper for storing and retrieving conversation messages
"""
import sqlite3
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationDB:
    """Manages conversation storage in SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the messages table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'agent')),
                    content TEXT NOT NULL,
                    tools_used TEXT,
                    skills_applied TEXT,
                    trajectory_id TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON messages(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON messages(timestamp)
            """)
            
            conn.commit()
            conn.close()
            logger.info("Conversation database initialized")
        except Exception as e:
            logger.error(f"Error initializing conversation DB: {e}")
            raise
    
    def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        tools_used: Optional[List[str]] = None,
        skills_applied: Optional[List[str]] = None,
        trajectory_id: Optional[str] = None
    ):
        """Save a message to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            tools_str = json.dumps(tools_used) if tools_used else None
            skills_str = json.dumps(skills_applied) if skills_applied else None
            
            cursor.execute("""
                INSERT INTO messages 
                (session_id, user_id, role, content, tools_used, skills_applied, trajectory_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, role, content, tools_str, skills_str, trajectory_id, timestamp))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved {role} message for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            raise
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve conversation history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT id, session_id, user_id, role, content, tools_used, 
                       skills_applied, trajectory_id, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()
            conn.close()
            
            messages = []
            for row in rows:
                msg = {
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'role': row['role'],
                    'content': row['content'],
                    'tools_used': json.loads(row['tools_used']) if row['tools_used'] else [],
                    'skills_applied': json.loads(row['skills_applied']) if row['skills_applied'] else [],
                    'trajectory_id': row['trajectory_id'],
                    'timestamp': row['timestamp']
                }
                messages.append(msg)
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def get_all_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all sessions with their last activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                query = """
                    SELECT session_id, user_id, 
                           MAX(timestamp) as last_activity,
                           COUNT(*) as message_count
                    FROM messages
                    WHERE user_id = ?
                    GROUP BY session_id, user_id
                    ORDER BY last_activity DESC
                """
                cursor.execute(query, (user_id,))
            else:
                query = """
                    SELECT session_id, user_id, 
                           MAX(timestamp) as last_activity,
                           COUNT(*) as message_count
                    FROM messages
                    GROUP BY session_id, user_id
                    ORDER BY last_activity DESC
                """
                cursor.execute(query)
            
            rows = cursor.fetchall()
            conn.close()
            
            sessions = []
            for row in rows:
                sessions.append({
                    'session_id': row['session_id'],
                    'user_id': row['user_id'],
                    'last_activity': row['last_activity'],
                    'message_count': row['message_count']
                })
            
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []

