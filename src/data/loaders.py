import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Structured representation of a conversation message"""
    user_input: str
    bot_response: str
    timestamp: datetime
    session_id: str
    message_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationLoader:
    """Enhanced conversation loader with validation and caching"""

    def __init__(self, data_path: Union[str, Path], cache_enabled: bool = True):
        self.data_path = Path(data_path)
        self.cache_enabled = cache_enabled
        self._cache = {}

    def load_conversations(self) -> List[ConversationMessage]:
        """Load and validate conversations from JSON files"""
        conversations = []

        if not self.data_path.exists():
            logger.error(f"Data path {self.data_path} does not exist")
            return conversations

        json_files = list(self.data_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} conversation files")

        for file_path in json_files:
            try:
                session_conversations = self._load_single_file(file_path)
                conversations.extend(session_conversations)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(conversations)} total conversations")
        return conversations

    def _load_single_file(self, file_path: Path) -> List[ConversationMessage]:
        """Load conversations from a single JSON file - adapted for your exact format"""
        cache_key = str(file_path)

        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        session_id = file_path.stem
        conversations = []

        # Handle your exact JSON structure
        for idx, exchange in enumerate(data):
            if not self._validate_exchange(exchange):
                logger.warning(f"Invalid exchange in {file_path} at index {idx}")
                continue

            message = ConversationMessage(
                user_input=exchange.get('user_input', ''),
                bot_response=exchange.get('bot_response', ''),
                timestamp=self._parse_timestamp(exchange.get('timestamp')),
                session_id=session_id,
                message_id=f"{session_id}_{idx}",
                metadata={}
            )
            conversations.append(message)

        if self.cache_enabled:
            self._cache[cache_key] = conversations

        return conversations

    def _validate_exchange(self, exchange: Dict[str, Any]) -> bool:
        """Validate a single conversation exchange - adapted for your format"""
        required_fields = ['user_input', 'bot_response', 'timestamp']
        return all(field in exchange and exchange[field] for field in required_fields)

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object - flexible parsing"""
        if not timestamp_str:
            return datetime.now()

        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            try:
                # Try other common formats
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
                except:
                    logger.warning(f"Could not parse timestamp: {timestamp_str}")
                    return datetime.now()


    def to_dataframe(self, conversations: List[ConversationMessage]) -> pd.DataFrame:
        """Convert conversations to pandas DataFrame"""
        data = []
        for conv in conversations:
            data.append({
                'user_input': conv.user_input,
                'bot_response': conv.bot_response,
                'timestamp': conv.timestamp,
                'session_id': conv.session_id,
                'message_id': conv.message_id,
                'metadata': conv.metadata
            })
        return pd.DataFrame(data)

class QuestionnaireLoader:
    """Loader for VARK, MBTI, and other questionnaire data"""

    def __init__(self, questionnaire_path: Union[str, Path]):
        self.questionnaire_path = Path(questionnaire_path)

    def load_vark_responses(self) -> pd.DataFrame:
        """Load VARK learning style questionnaire responses"""
        try:
            with open(self.questionnaire_path / "vark_responses.json", 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except FileNotFoundError:
            logger.warning("VARK responses file not found")
            return pd.DataFrame()

    def load_mbti_responses(self) -> pd.DataFrame:
        """Load MBTI personality questionnaire responses"""
        try:
            with open(self.questionnaire_path / "mbti_responses.json", 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except FileNotFoundError:
            logger.warning("MBTI responses file not found")
            return pd.DataFrame()
