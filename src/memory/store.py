"""
Memory and Storage Module

This module handles:
- Storing user profiles in local memory
- Managing consent flows
- Storing consented submissions to bank database
- Audit logging
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/actions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/consented_submissions", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_memory() -> Dict[str, Any]:
    """
    Load user memory from disk.
    
    Returns:
        Dictionary keyed by user_id containing profiles and conversation logs
    """
    memory_path = "data/memory.json"
    
    if not os.path.exists(memory_path):
        logger.info("Memory file does not exist. Creating new memory.")
        return {}
    
    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        logger.info(f"Loaded memory with {len(memory)} users")
        return memory
    except Exception as e:
        logger.error(f"Error loading memory: {e}")
        return {}


def save_memory(memory: Dict[str, Any]):
    """
    Save user memory to disk.
    
    Args:
        memory: Memory dictionary to save
    """
    memory_path = "data/memory.json"
    
    try:
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved memory with {len(memory)} users")
    except Exception as e:
        logger.error(f"Error saving memory: {e}")
        raise


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user profile from memory.
    
    Args:
        user_id: User identifier
        
    Returns:
        User profile dictionary or None if not found
    """
    memory = load_memory()
    return memory.get(user_id, {}).get("profile")


def save_user_profile(user_id: str, profile: Dict[str, Any], 
                     conversation_log: Optional[List[Dict[str, Any]]] = None):
    """
    Save user profile to memory.
    
    Args:
        user_id: User identifier
        profile: User profile dictionary
        conversation_log: Optional conversation log list
    """
    memory = load_memory()
    
    if user_id not in memory:
        memory[user_id] = {
            "profile": profile,
            "conversation_log": conversation_log or [],
            "created_at": datetime.now().isoformat()
        }
    else:
        memory[user_id]["profile"] = profile
        if conversation_log is not None:
            memory[user_id]["conversation_log"] = conversation_log
        memory[user_id]["updated_at"] = datetime.now().isoformat()
    
    save_memory(memory)
    logger.info(f"Saved profile for user {user_id}")


def delete_user_data(user_id: str):
    """
    Delete all user data from memory (when consent is denied).
    
    Args:
        user_id: User identifier
    """
    memory = load_memory()
    
    if user_id in memory:
        del memory[user_id]
        save_memory(memory)
        logger.info(f"Deleted all data for user {user_id}")
    else:
        logger.warning(f"User {user_id} not found in memory")


def save_consented_submission(user_id: str, profile: Dict[str, Any], 
                              decision: Dict[str, Any]):
    """
    Save consented submission to bank database.
    
    Args:
        user_id: User identifier
        profile: User profile dictionary
        decision: RAG decision dictionary
    """
    timestamp = datetime.now().isoformat()
    
    # Save to consented_submissions folder
    submission_file = f"data/consented_submissions/{user_id}_{timestamp.replace(':', '-')}.json"
    
    submission = {
        "user_id": user_id,
        "timestamp": timestamp,
        "profile": profile,
        "decision": decision
    }
    
    try:
        with open(submission_file, 'w', encoding='utf-8') as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved consented submission to {submission_file}")
    except Exception as e:
        logger.error(f"Error saving consented submission: {e}")
        raise
    
    # Append to bank unstructured database (JSONL format)
    bank_db_path = "data/bank_unstructured_db.jsonl"
    
    try:
        with open(bank_db_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(submission, ensure_ascii=False) + "\n")
        logger.info(f"Appended to bank database: {bank_db_path}")
    except Exception as e:
        logger.error(f"Error appending to bank database: {e}")
        raise


def get_bank_submissions(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all submissions from bank database (for admin view).
    
    Args:
        limit: Optional limit on number of submissions to return
        
    Returns:
        List of submission dictionaries
    """
    bank_db_path = "data/bank_unstructured_db.jsonl"
    
    if not os.path.exists(bank_db_path):
        logger.info("Bank database does not exist yet")
        return []
    
    submissions = []
    try:
        with open(bank_db_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    submissions.append(json.loads(line))
        
        # Reverse to show most recent first
        submissions.reverse()
        
        if limit:
            submissions = submissions[:limit]
        
        logger.info(f"Retrieved {len(submissions)} submissions from bank database")
        return submissions
    except Exception as e:
        logger.error(f"Error reading bank database: {e}")
        return []


def log_action(action: str, user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """
    Log an action for audit purposes.
    
    Args:
        action: Action description
        user_id: Optional user identifier
        details: Optional additional details
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "user_id": user_id,
        "details": details or {}
    }
    
    logger.info(f"Action: {action}, User: {user_id}, Details: {details}")


