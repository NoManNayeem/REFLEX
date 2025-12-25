"""
Security utilities for input validation and sanitization
"""

import re
import html
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input exceeds max length
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        if len(text) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        # HTML escape for safety (backend sanitization)
        # Note: Frontend should also sanitize with DOMPurify
        text = html.escape(text)
        
        return text.strip()
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format and safety
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid and safe, False otherwise
        """
        # Check basic format
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL protocol: {url}")
            return False
        
        # Block local/private addresses to prevent SSRF
        blocked_patterns = [
            r'localhost',
            r'127\.0\.0\.',
            r'192\.168\.',
            r'10\.',
            r'172\.(1[6-9]|2[0-9]|3[01])\.',
            r'::1',
            r'0\.0\.0\.0'
        ]
        
        for pattern in blocked_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                logger.warning(f"Blocked private/local URL: {url}")
                return False
        
        # Check URL length
        if len(url) > 2048:
            logger.warning(f"URL too long: {len(url)} chars")
            return False
        
        return True
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """
        Validate session ID format
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Session IDs should be alphanumeric with underscores/hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            logger.warning(f"Invalid session ID format: {session_id}")
            return False
        
        # Reasonable length limits
        if len(session_id) < 3 or len(session_id) > 128:
            logger.warning(f"Invalid session ID length: {len(session_id)}")
            return False
        
        return True
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """
        Validate user ID format
        
        Args:
            user_id: User ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Similar to session ID validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            logger.warning(f"Invalid user ID format: {user_id}")
            return False
        
        if len(user_id) < 3 or len(user_id) > 128:
            logger.warning(f"Invalid user ID length: {len(user_id)}")
            return False
        
        return True
